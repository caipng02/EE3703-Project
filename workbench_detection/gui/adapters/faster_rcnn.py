"""Faster R-CNN in the GUI: the run's best checkpoint with the confidence and RPN NMS IoU chosen on validation."""

from pathlib import Path

READY = False
STATUS = "No Faster R-CNN run connected. Start with: python -m workbench_detection gui --frcnn-run runs/<name>"
detector = None


def connect(run_dir, device="cpu"):
    """Load the run's final checkpoint if the test has been run, otherwise its development checkpoint."""
    global READY, STATUS, detector
    from ...engine.faster_rcnn import FasterRCNNDetector, FasterRCNNRun
    from ...evaluation.records import load_json

    run = FasterRCNNRun(run_dir)
    checkpoint = run.best_checkpoint()
    detector = FasterRCNNDetector.from_checkpoint(checkpoint, device, load_json(run.protocol_path))
    READY = True
    STATUS = f"Faster R-CNN loaded: {Path(run_dir).name} ({checkpoint.parent.name})"


def predict(image):
    if not READY:
        raise RuntimeError(STATUS)
    return detector.predict_image(image)
