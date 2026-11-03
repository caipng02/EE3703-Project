"""Faster R-CNN in the GUI. In progress: connect it the same way as adapters/yolo.py.

predict(image) receives a PIL RGB photo of any size and returns
[{"label": "drill" | "scissor" | "screwdriver", "confidence": float, "box": [x1, y1, x2, y2]}]
in the photo's own pixels, after the same preprocessing, thresholds and NMS as the final evaluation.
"""

READY = False
STATUS = "Faster R-CNN is not connected yet."


def predict(image):
    raise RuntimeError(STATUS)
