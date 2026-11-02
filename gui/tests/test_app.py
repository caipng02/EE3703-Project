"""Integration checks for the upload / detector boundary, without model weights."""

import base64
import http.client
import io
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import ModelRegistry, create_server, decode_image, validate_detections
from adapters.preprocessing import letterbox, restore_box


def encoded_image(size=(320, 160), mode="RGB", color="white"):
    output = io.BytesIO()
    Image.new(mode, size, color).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = ModelRegistry()
        cls.server = create_server(0, cls.registry)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        body = json.dumps(payload) if payload is not None else None
        connection.request(method, path, body, headers or {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        content_type = response.getheader("Content-Type")
        connection.close()
        return response.status, json.loads(data) if "json" in content_type else data

    def test_page_and_static_assets_load(self):
        for path in ["/", "/app.js", "/styles.css", "/favicon.svg"]:
            code, body = self.request("GET", path)
            self.assertEqual(code, 200)
            self.assertTrue(body)
        self.assertEqual(self.request("GET", "/../server.py")[0], 404)

    def test_both_adapters_start_unavailable_and_do_not_fabricate_detections(self):
        code, body = self.request("GET", "/api/models")
        self.assertEqual(code, 200)
        self.assertEqual([m["ready"] for m in body["models"]], [False, False])
        for model in ["yolo", "faster_rcnn"]:
            code, body = self.request("POST", "/api/predict", {"model": model, "image": encoded_image()})
            self.assertEqual(code, 503)
            self.assertIn("not connected", body["error"])
            self.assertNotIn("detections", body)

    def test_uploaded_pixels_and_selected_model_reach_the_adapter(self):
        original = self.registry.modules["yolo"]
        seen = []

        def predict(image):
            seen.append((image.mode, image.size, image.getpixel((0, 0))))
            return [{"label": "drill", "confidence": 0.93, "box": [10, 20, 100, 120]}]

        self.registry.modules["yolo"] = SimpleNamespace(READY=True, STATUS="Test only", predict=predict)
        try:
            code, body = self.request("POST", "/api/predict", {"model": "yolo", "image": encoded_image(color="red")})
            self.assertEqual(code, 200)
            self.assertEqual(seen, [("RGB", (320, 160), (255, 0, 0))])
            self.assertEqual(body["detections"][0]["label"], "drill")
            self.assertEqual(body["detections"][0]["box"], [10, 20, 100, 120])
            self.assertEqual((body["width"], body["height"]), (320, 160))
            self.assertGreaterEqual(body["inference_ms"], 0)
            self.assertEqual(decode_image(body["image"]).size, (320, 160))
        finally:
            self.registry.modules["yolo"] = original

    def test_rejects_bad_uploads_unknown_models_and_cross_origin_requests(self):
        for payload in [{}, [], {"model": "yolo", "image": "%%%"},
                        {"model": "yolo", "image": base64.b64encode(b"not an image").decode()},
                        {"model": "other", "image": encoded_image()}]:
            self.assertEqual(self.request("POST", "/api/predict", payload)[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", {},
                         {"Content-Type": "application/json", "Origin": "https://example.org"})[0], 403)
        self.assertEqual(self.request("GET", "/", headers={"Host": "example.org"})[0], 403)

    def test_model_output_cannot_use_wrong_coordinate_space_or_nonfinite_values(self):
        for confidence, box in [(float("nan"), [0, 0, 10, 10]), (1.1, [0, 0, 10, 10]),
                                (0.8, [0, 0, 640, 640]), (0.8, [10, 10, 0, 0])]:
            with self.assertRaises(ValueError):
                validate_detections([{"label": "drill", "confidence": confidence, "box": box}], (320, 160))
        self.assertEqual(validate_detections([], (320, 160)), [])

    def test_portrait_letterbox_roundtrip_accounts_for_rounding_and_padding(self):
        image = Image.new("RGB", (333, 777))
        padded, transform = letterbox(image)
        self.assertEqual(padded.size, (640, 640))
        original = [12, 30, 300, 720]
        mapped = [original[0] * transform["scale_x"] + transform["left"],
                  original[1] * transform["scale_y"] + transform["top"],
                  original[2] * transform["scale_x"] + transform["left"],
                  original[3] * transform["scale_y"] + transform["top"]]
        for actual, expected in zip(restore_box(mapped, transform), original):
            self.assertAlmostEqual(actual, expected)

    def test_transparent_pixels_are_composited_on_white(self):
        image = decode_image(encoded_image(mode="RGBA", color=(0, 0, 0, 0)))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
