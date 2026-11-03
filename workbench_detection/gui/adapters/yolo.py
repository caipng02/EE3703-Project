"""YOLOv2 in the GUI: the run's best checkpoint with the confidence and NMS IoU chosen on validation."""

from pathlib import Path

READY = False
STATUS = "No YOLOv2 run connected. Start with: python -m workbench_detection gui --run runs/<name>"
detector = None


def connect(run_dir, device="cpu"):
    """Load the run's final checkpoint if the test has been run, otherwise its development checkpoint."""
    global READY, STATUS, detector
    from ...engine.predictor import YOLOv2Detector
    from ...engine.workflow import Run
    from ...evaluation.records import load_json

    run = Run(run_dir)
    checkpoint = run.best_checkpoint()
    detector = YOLOv2Detector.from_checkpoint(checkpoint, device, load_json(run.protocol_path))
    READY = True
    STATUS = f"YOLOv2 loaded: {Path(run_dir).name} ({checkpoint.parent.name})"


def predict(image):
    if not READY:
        raise RuntimeError(STATUS)
    return detector.predict_image(image)
