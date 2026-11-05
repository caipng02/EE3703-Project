"""Inference: one detector object used by validation, the final test, the GUI and the notebook demo.

Detections are pixel xyxy boxes on the 640 x 640 canvas with class IDs and scores, after class-wise NMS.
For photos of any size, predict_image letterboxes first and maps the boxes back to the photo's pixels.
"""

import numpy as np
import torch

from ..config import AP_SCORE_FLOOR, CLASSES, IMAGE_SIZE, MODEL_NAME, TrainConfig
from ..data.dataset import normalize
from ..data.transforms import letterbox, restore_box
from ..evaluation.records import predictions_from_outputs
from ..models.loss import yolo_loss_on_cpu
from ..models.yolov2 import YOLOv2ResNet18, postprocess


# %% predictor
def load_checkpoint(path, device):
    """Model in eval mode plus the checkpoint dict (anchors, config, stage, protocol)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order differs from the dataset.")
    TrainConfig.from_dict(checkpoint["config"])  # refuses checkpoints of a different architecture
    model = YOLOv2ResNet18(checkpoint["anchors"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


class YOLOv2Detector:
    """A trained YOLOv2 model with its operating settings (confidence threshold and NMS IoU)."""

    name, family = MODEL_NAME, "yolov2"

    def __init__(self, model, device, *, confidence=0.25, nms_iou=0.5):
        self.model, self.device = model, torch.device(device)
        self.confidence, self.nms_iou = confidence, nms_iou

    @classmethod
    def from_checkpoint(cls, path, device, protocol=None):
        """Load a checkpoint; with a validation protocol, use the confidence and NMS IoU chosen on validation."""
        model, _ = load_checkpoint(path, device)
        if protocol is None:
            return cls(model, device)
        settings = protocol["models"][MODEL_NAME]
        return cls(model, device, confidence=settings["operating_confidence"], nms_iou=settings["nms"]["final_nms_iou"])

    def decode(self, raw, score_floor, nms_iou):
        """Raw outputs -> per-image detections in canvas pixels. NMS runs on the CPU, where its loop is faster."""
        results = postprocess(raw.detach().cpu(), self.model.anchors.cpu(), conf=score_floor, nms_iou=nms_iou)
        for row in results:
            row["boxes"] *= IMAGE_SIZE
        return results

    @torch.inference_mode()
    def detect(self, images, score_floor=None, nms_iou=None):
        """Normalized image batch (B, 3, 640, 640) -> detections, at the operating settings by default."""
        raw = self.model(images.to(self.device)).float()
        return self.decode(raw, self.confidence if score_floor is None else score_floor,
                           self.nms_iou if nms_iou is None else nms_iou)

    def predict_image(self, image):
        """A PIL RGB photo of any size -> [{"label", "confidence", "box"}] in the photo's own pixels."""
        canvas, transform = letterbox(image)
        result = self.detect(normalize(np.array(canvas))[None])[0]
        detections = []
        for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
            restored = restore_box(box, transform)
            if restored[2] > restored[0] and restored[3] > restored[1]:
                detections.append({"label": CLASSES[label], "confidence": score, "box": restored})
        return detections

    @torch.inference_mode()
    def export(self, loader, ground_truth, nms_ious, score_floor=AP_SCORE_FLOOR, checkpoint="", with_loss=False):
        """Predictions for a whole split, one export per NMS IoU, in the evaluator's format.

        Exports keep every detection above score_floor so AP sees the full precision-recall curve.
        With with_loss, also returns the mean YOLOv2 loss (validation loss during training).
        """
        self.model.eval()
        targets, collected = [], {iou: [] for iou in nms_ious}
        total_loss = 0.0
        for images, rows in loader:
            raw = self.model(images.to(self.device)).float()
            if with_loss:
                batch_loss, _ = yolo_loss_on_cpu(raw, rows, self.model.anchors)
                total_loss += batch_loss.item() * len(rows)
            for iou in nms_ious:
                collected[iou].extend(self.decode(raw, score_floor, iou))
            targets.extend(rows)
        exports = [predictions_from_outputs(
            targets, collected[iou], ground_truth, self.name,
            label_mapping=dict(enumerate(range(len(CLASSES)))),
            metadata={"checkpoint": str(checkpoint), "model_family": self.family, "nms_stage": "final",
                      "nms_iou": iou, "nms_rule": "class_aware", "export_score_floor": score_floor},
        ) for iou in nms_ious]
        return exports, (total_loss / len(targets) if with_loss else None)

