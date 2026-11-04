"""Local web interface: upload a photo, choose YOLOv2 or Faster R-CNN, see boxes, labels and confidences.

Start it with python -m workbench_detection gui --run runs/<name>. Images are processed in memory only.
"""

import base64
import binascii
import importlib
import io
import json
import logging
import math
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_BODY_BYTES = 17 * 1024 * 1024
MAX_PIXELS = 24_000_000
CLASSES = ("drill", "scissor", "screwdriver")
MODEL_NAMES = {"yolo": "YOLO", "faster_rcnn": "Faster R-CNN"}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}


class InputError(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def decode_image(encoded: str) -> Image.Image:
    if not isinstance(encoded, str):
        raise InputError("Choose a JPG, PNG or WebP image.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InputError("The uploaded image could not be read.") from error
    if len(data) > MAX_IMAGE_BYTES:
        raise InputError("Please choose an image smaller than 12 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InputError("Choose a JPG, PNG or WebP image.")
                if source.width * source.height > MAX_PIXELS:
                    raise InputError("Please choose an image with at most 24 megapixels.")
                image = ImageOps.exif_transpose(source).convert("RGBA")
                background = Image.new("RGBA", image.size, "white")
                return Image.alpha_composite(background, image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise InputError("This image is damaged or too large. Try another image.") from error


def validate_detections(predictions, size: tuple[int, int]) -> list[dict]:
    """Enforce the adapter contract before drawing anything in the browser."""
    if not isinstance(predictions, list) or len(predictions) > 500:
        raise ValueError("predict() must return a list with at most 500 detections.")
    width, height = size
    checked = []
    for detection in predictions:
        if not isinstance(detection, dict) or detection.get("label") not in CLASSES:
            raise ValueError("Detection labels must be drill, scissor or screwdriver.")
        confidence = detection.get("confidence")
        box = detection.get("box")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Confidence must be a finite number between 0 and 1.")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in box)):
            raise ValueError("Boxes must contain four finite pixel coordinates.")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Boxes must be xyxy pixels inside the original image.")
        checked.append({"label": detection["label"], "confidence": float(confidence),
                        "box": list(map(float, box))})
    return sorted(checked, key=lambda item: item["confidence"], reverse=True)


class ModelRegistry:
    def __init__(self):
        self.modules = {}
        self.errors = {}
        self.lock = threading.Lock()
        for key in MODEL_NAMES:
            try:
                self.modules[key] = importlib.import_module(f".adapters.{key}", __package__)
            except Exception:
                logging.exception("Could not import %s adapter", key)
                self.errors[key] = "Adapter could not load. Check the server terminal."

    def status(self) -> list[dict]:
        return [{"id": key, "name": name,
                 "ready": key in self.modules and bool(getattr(self.modules[key], "READY", False)),
                 "message": self.errors.get(key, getattr(self.modules.get(key), "STATUS", "Not connected"))}
                for key, name in MODEL_NAMES.items()]

    def predict(self, key: str, image: Image.Image) -> tuple[list[dict], float]:
        if key not in MODEL_NAMES:
            raise InputError("Choose YOLO or Faster R-CNN.")
        module = self.modules.get(key)
        if module is None or not getattr(module, "READY", False):
            raise ModelUnavailable(f"{MODEL_NAMES[key]} is not connected yet. "
                                   "Connect its trained model adapter and restart the app.")
        with self.lock:
            start = time.perf_counter()
            predictions = module.predict(image.copy())
            elapsed = (time.perf_counter() - start) * 1000
            return validate_detections(predictions, image.size), elapsed


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, code: int, content: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; "
                         "style-src 'self'; script-src 'self'; connect-src 'self'; "
                         "frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, code: int, value: dict):
        self.send_bytes(code, json.dumps(value, allow_nan=False).encode(), "application/json")

    def valid_host(self) -> bool:
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_GET(self):
        if not self.valid_host():
            return self.send_json(403, {"error": "Open this app using its localhost address."})
        path = self.path.split("?", 1)[0]
        if path == "/api/models":
            return self.send_json(200, {"models": self.server.registry.status()})
        if path in STATIC_FILES:
            name, content_type = STATIC_FILES[path]
            return self.send_bytes(200, (ROOT / "static" / name).read_bytes(), content_type)
        self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        if not self.valid_host():
            return self.send_json(403, {"error": "Open this app using its localhost address."})
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.send_json(403, {"error": "Upload images from the app page."})
        if self.path != "/api/predict":
            return self.send_json(404, {"error": "Page not found."})
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise InputError("The upload could not be read. Please try again.") from error
            if not 0 < length <= MAX_BODY_BYTES:
                return self.send_json(413, {"error": "Please choose an image smaller than 12 MB."})
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                raise InputError("Upload a supported image using the app.")
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise InputError("The upload could not be read. Please try again.") from error
            if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
                raise InputError("Choose a detector and upload an image.")
            image = decode_image(payload.get("image"))
            predictions, elapsed = self.server.registry.predict(payload["model"], image)
            output = io.BytesIO()
            image.save(output, format="PNG")
            self.send_json(200, {"model": payload["model"], "detections": predictions,
                                 "inference_ms": round(elapsed, 1),
                                 "width": image.width, "height": image.height,
                                 "image": base64.b64encode(output.getvalue()).decode()})
        except InputError as error:
            self.send_json(400, {"error": str(error)})
        except ModelUnavailable as error:
            self.send_json(503, {"error": str(error)})
        except Exception:
            logging.exception("Detection request failed")
            self.send_json(500, {"error": "Detection failed. Check the server terminal, then try again."})


def create_server(port=8765, registry=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.registry = registry if registry is not None else ModelRegistry()
    return server


def serve(port=8765, *, run=None, frcnn_run=None, device="cpu"):
    """Connect each adapter to its run's best checkpoint (if given) and serve until Ctrl+C."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if run is not None:
        from .adapters import yolo
        yolo.connect(run, device)
    if frcnn_run is not None:
        from .adapters import faster_rcnn
        faster_rcnn.connect(frcnn_run, device)
    with create_server(port) as server:
        print(f"\nWorkbench is ready: http://127.0.0.1:{server.server_port}\n", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nWorkbench stopped.")
