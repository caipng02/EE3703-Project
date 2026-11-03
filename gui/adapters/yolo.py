"""Connect the browser to the same YOLOv2 checkpoint and operating settings."""

import os
from pathlib import Path

READY = False
STATUS = "Set WORKBENCH_YOLO_CHECKPOINT and WORKBENCH_YOLO_PROTOCOL to connect YOLOv2."

checkpoint = os.environ.get("WORKBENCH_YOLO_CHECKPOINT")
protocol_path = os.environ.get("WORKBENCH_YOLO_PROTOCOL")

if checkpoint and protocol_path:
    import torch
    from integrate_yolo import MODEL_NAME, load_checkpoint, predict_uploaded_image
    from person4_evaluation import load_json
    from .preprocessing import letterbox, restore_box

    protocol = load_json(protocol_path)
    policy = protocol["models"][MODEL_NAME]
    device = torch.device(os.environ.get("WORKBENCH_YOLO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, saved = load_checkpoint(Path(checkpoint), device)
    if saved.get("stage") == "final" and saved.get("protocol") != protocol:
        raise ValueError("YOLO checkpoint and frozen protocol differ.")
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    READY = True
    STATUS = "YOLOv2 checkpoint loaded."


def predict(image):
    if not READY:
        raise RuntimeError(STATUS)
    return predict_uploaded_image(model, image, device, policy["operating_confidence"],
                                  policy["nms"]["final_nms_iou"], letterbox, restore_box)
