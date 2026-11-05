"""The experiment workflow: software check, development, validation, then one final test.

A run lives in one folder:
    runs/<name>/config.yaml                      the training settings
    runs/<name>/work/<stage>/                    checkpoints, history.csv, evaluation_protocol.json
    runs/<name>/report/<stage>/                  metrics.txt, figures and example predictions
Stages: train, validation, final_train, test.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset

from ..config import AP_SCORE_FLOOR, MODEL_NAME, TrainConfig, save_config, seed_everything
from ..data.dataset import ToolDataset, normalize
from ..evaluation.benchmark import benchmark_inference, model_size_summary
from ..evaluation.metrics import EvaluationConfig
from ..evaluation.protocol import NMS_GRIDS, run_evaluation
from ..evaluation.records import load_ground_truth, load_json, save_json
from ..models.loss import encode_targets
from ..models.yolov2 import YOLOv2ResNet18, anchors_from_dataset
from .predictor import YOLOv2Detector, load_checkpoint
from .trainer import make_loader, make_scaler, train, training_step


# %% run
class Run:
    """Paths of one experiment run."""

    model_name = MODEL_NAME  # the model's key in the run's evaluation protocol

    def __init__(self, root):
        self.root = Path(root)

    def work(self, stage):
        """Folder for a stage's checkpoints and protocol files."""
        return self.root / "work" / stage

    def report(self, stage):
        """Folder for a stage's metrics, figures and result.json."""
        return self.root / "report" / stage

    @property
    def protocol_path(self):
        """The validation protocol: settings frozen for the test."""
        return self.work("validation") / "evaluation_protocol.json"

    def development_checkpoint(self):
        """The checkpoint validation selected settings for (recorded in the protocol).

        The protocol records an absolute path (e.g. on Colab's Google Drive). If the run folder has since been moved
        or downloaded, the same file is looked up inside this run folder instead.
        """
        recorded = Path(load_json(self.protocol_path)["models"][self.model_name]["validation_checkpoint"])
        if not recorded.is_file() and self.root.name in recorded.parts:
            index = len(recorded.parts) - 1 - recorded.parts[::-1].index(self.root.name)
            moved = self.root.joinpath(*recorded.parts[index + 1:])
            if moved.is_file():
                return moved
        return recorded

    def final_checkpoint(self):
        """The model retrained on train + validation for the test."""
        return self.work("final_train") / "best.pt"

    def best_checkpoint(self):
        """The final checkpoint once the test has been run, otherwise the development checkpoint."""
        return self.final_checkpoint() if self.final_checkpoint().is_file() else self.development_checkpoint()

    def trained_checkpoint(self):
        """The development checkpoint if training has finished in this run, else None."""
        checkpoint = self.work("train") / "best.pt"
        return checkpoint if (self.work("train") / "history.csv").is_file() else None

    def result(self, stage):
        """The saved validation or test result, or None if that stage has not been run."""
        path = self.report(stage) / "result.json"
        return load_json(path) if path.is_file() else None

    def ensure_fresh(self, *stages):
        """Refuse to overwrite stages that already have results."""
        for stage in stages:
            for folder in (self.work(stage), self.report(stage)):
                if folder.exists() and any(folder.iterdir()):
                    raise FileExistsError(f"{folder} already has results; use a new run name.")


# %% stages
def software_check(dataset_root, out_dir, device, batch=2):
    """Software check on the 8 contract images: one training update, decoding and scoring.

    Confirms that boxes, class IDs, the model and the evaluator connect. Scores are not detector results.
    """
    seed_everything()
    dataset_root = Path(dataset_root)
    ground_truth = load_ground_truth(dataset_root, "contract")
    names = {image["image_id"] for image in ground_truth["images"]}
    training = ToolDataset(dataset_root, "train")
    subset = Subset(training, [i for i, name in enumerate(training.names) if name in names])
    loader = make_loader(subset, batch)
    images, targets = next(iter(loader))
    model = YOLOv2ResNet18(anchors_from_dataset(training)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = training_step(model, optimizer, images, targets, device, make_scaler(device))
    print(f"Training batch {tuple(images.shape)} | one-update loss={loss:.4f}")
    print(f"First image {targets[0]['file_name']} | targets [class, cx, cy, w, h]:")
    print(encode_targets(targets)[0])
    predictions, _ = YOLOv2Detector(model, device).export(loader, ground_truth, [0.50], AP_SCORE_FLOOR,
                                                          checkpoint="inspection")
    for row in predictions[0]["images"]:  # keep the display readable while the model is untrained
        for field in ("boxes", "labels", "scores"):
            row[field] = row[field][:5]
    predictions[0]["metadata"]["prediction_kind"] = "software_fixture"
    run_evaluation(ground_truth, predictions, out_dir,
                   EvaluationConfig(ap_backend="coco", ap_score_floor=AP_SCORE_FLOOR, operating_confidence=AP_SCORE_FLOOR),
                   split="contract", image_root=dataset_root / "contract" / "images", evaluation_purpose="software_check")


def develop(config, dataset_root, run, device):
    """Development training; returns the checkpoint with the best validation AP50."""
    run.ensure_fresh("train")
    save_config(config, run.root / "config.yaml")
    return train(config, dataset_root, run.work("train"), run.report("train"), device)["checkpoint"]


def evaluate_split(checkpoint, dataset_root, split, work_dir, report_dir, device, *, batch=8, protocol=None,
                   benchmark=False):
    """Score a checkpoint on validation (choosing NMS and confidence) or on the test (frozen protocol)."""
    final = split == "test"
    dataset_root, checkpoint = Path(dataset_root), Path(checkpoint)
    model, saved = load_checkpoint(checkpoint, device)
    if final and (saved.get("stage") != "final" or saved.get("protocol") != protocol):
        raise ValueError("The test requires the final checkpoint retrained with this frozen protocol.")
    if not final and saved.get("stage") == "final":
        raise ValueError("Select settings on validation with the development checkpoint.")
    detector = YOLOv2Detector(model, device)
    floor = protocol["ap_score_floor"] if final else AP_SCORE_FLOOR
    candidates = [protocol["models"][MODEL_NAME]["nms"]["nms_iou"]] if final else NMS_GRIDS["yolov2"]
    ground_truth = load_ground_truth(dataset_root, split, allow_test=final)
    image_root = dataset_root / split / "images"
    loader = make_loader(ToolDataset(dataset_root, split, allow_test=final), batch)
    predictions, _ = detector.export(loader, ground_truth, candidates, floor, checkpoint=checkpoint.resolve())
    size = model_size_summary(model, checkpoint)
    for prediction in predictions:
        prediction["metadata"]["model_size"] = size
    if benchmark:
        samples = []
        for image in ground_truth["images"]:  # decoded before timing; preprocessing is timed
            with Image.open(image_root / image["file_name"]) as opened:
                samples.append(np.array(opened.convert("RGB")))
        settings = protocol["models"][MODEL_NAME]
        predictions[0]["metadata"]["benchmark"] = benchmark_inference(
            lambda image: detector.detect(normalize(image)[None], settings["operating_confidence"], candidates[0]),
            samples, device=str(device), precision="float32",
            synchronize=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None)
    return run_evaluation(ground_truth, predictions, work_dir, EvaluationConfig(ap_score_floor=floor),
                          split=split, image_root=image_root, project_protocol=True, select_nms=not final,
                          frozen_protocol=protocol, results_dir=report_dir)[0]


def validate(checkpoint, dataset_root, run, device, batch=8):
    """Choose NMS IoU and operating confidence on validation; writes the protocol the test will use.

    Returns the validation result (metrics, per-class rows, error records) for the selected settings.
    """
    run.ensure_fresh("validation")
    result = evaluate_split(checkpoint, dataset_root, "valid", run.work("validation"), run.report("validation"),
                            device, batch=batch)
    save_json(run.report("validation") / "result.json", result)
    return result


def final_test(run, dataset_root, device, batch=None):
    """Retrain on train + validation with the frozen settings, then evaluate once on the 50 test originals.

    A finished test (report/test/result.json) is never repeated. An interrupted one, stopped before its result was
    saved, is cleared and restarted; its retrain had not been evaluated, so nothing was learned from the test.
    """
    if run.result("test") is not None:
        raise FileExistsError(f"{run.root} already has a final test result; it is never repeated.")
    for stage in ("final_train", "test"):
        for folder in (run.work(stage), run.report(stage)):
            if folder.exists():
                print(f"Clearing an interrupted final test: {folder}", flush=True)
                shutil.rmtree(folder)
    if batch is None:  # evaluate with the run's own batch size, as validation did
        batch = TrainConfig.from_dict(torch.load(run.development_checkpoint(), map_location="cpu",
                                                 weights_only=True)["config"]).batch
    summary = train(None, dataset_root, run.work("final_train"), run.report("final_train"), device,
                    final_from=run.development_checkpoint(), protocol_path=run.protocol_path)
    result = evaluate_split(summary["checkpoint"], dataset_root, "test", run.work("test"), run.report("test"), device,
                            batch=batch, protocol=load_json(run.protocol_path), benchmark=True)
    save_json(run.report("test") / "result.json", result)
    return result
