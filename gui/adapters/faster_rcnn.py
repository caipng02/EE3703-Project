"""Connect the group's Faster R-CNN inference here; see ../README.md."""

READY = False
STATUS = "Waiting for trained model"


def predict(image):
    """Return detections for a PIL RGB image, using original-image pixel boxes.

    Return value: list of {"label": str, "confidence": float,
                           "box": [x1, y1, x2, y2]}.
    Labels: drill, scissor, screwdriver. Use the group's own trained detector.
    """
    raise RuntimeError("Connect your Faster R-CNN model before setting READY = True.")
