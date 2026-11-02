# Workbench — browser GUI

A local website for the EE3703 bonus demonstration. Upload an image, select
YOLO or Faster R-CNN, and view the detector's bounding boxes, class labels and
confidence scores directly on the website. You can switch between the original
and annotated image and see inference time. No download is required or offered.

**Current state:** the website is usable, but neither trained model is connected.
The adapters are intentionally disabled. No simulated detections, dataset labels
or pretrained substitute detectors are used as predictions.

## Start the website

From the repository root, using Python 3.10 or later:

```sh
python3 -m venv gui/.venv
gui/.venv/bin/python -m pip install -r gui/requirements.txt
gui/.venv/bin/python gui/server.py
```

Open **http://127.0.0.1:8765** in your browser. Stop the app with Ctrl+C in the
terminal. For a different port, use `python gui/server.py --port 8766` from an
environment with the requirements installed.

Only Pillow is needed for the website. Install your model's existing dependencies
(e.g. PyTorch) in the same environment when connecting the detectors. The app binds
to this computer only; no hosting account or internet connection is needed to run it.
Images are processed in memory and are not saved by the server.

## Connect the two trained models

Each model lead edits one file:

| Model | Connection point |
|---|---|
| YOLO / YOLOv2 | `adapters/yolo.py` |
| Faster R-CNN | `adapters/faster_rcnn.py` |

1. Import your actual model architecture and load your trained weights. Keep
   weights in `gui/weights/` (ignored by Git). Load the model once, rather than
   reloading it for every uploaded image.
2. Implement `predict(image)` in your adapter. The input is a Pillow RGB image,
   with EXIF orientation applied and transparent pixels composited onto white.
   It may be any size or aspect ratio. Do not assume it is already 640 × 640.
3. Apply exactly the preprocessing, decoding, score threshold and NMS used in
   your final evaluation. Return the detections in the format below. Convert
   NumPy/PyTorch values to ordinary Python strings, floats and lists.
4. Set `READY = True` and `STATUS = "Trained model connected"` once your adapter
   works. Restart the server and refresh the browser.

### Prediction contract

`predict(image)` must return a list. Each detection has these fields:

```python
# Format illustration only; these values must come from your actual model.
{
    "label": "drill",             # drill, scissor or screwdriver
    "confidence": 0.93,           # probability from 0.0 to 1.0
    "box": [42.0, 30.0, 210.0, 180.0],  # x1, y1, x2, y2
}
```

Boxes must be **pixel coordinates on the original uploaded image**, within its
boundaries, with x1 < x2 and y1 < y2. Return `[]` when no tools are detected.
There is a 500-detection limit per image.

Canonical class IDs in the dataset are `0 drill`, `1 scissor`, `2 screwdriver`.
COCO category IDs are one greater. Map your model's output IDs explicitly;
Faster R-CNN may also have a background class. The website displays `scissor`
as “Scissors”. Do not change class mappings based on the selected model.

`adapters/preprocessing.py` provides optional `letterbox()` and `restore_box()`
helpers for a 640 × 640 input. Use these only if their interpolation and fill
match your final pipeline. The helper records separate x/y scales to account for
rounding and clips restored boxes to the image; discard any boxes with no area
after clipping. Do not add letterboxing or normalization twice if the model
already handles it internally. Your training handoff uses ImageNet mean/std;
the adapter must follow what the actual trained detector expects.

For PyTorch models, use evaluation mode and inference/no-gradient mode. Perform
any necessary GPU synchronization inside the adapter before returning if you
want accurate GPU inference timing. The displayed time measures `predict()`
including its preprocessing and first-call initialization; it excludes upload,
JSON encoding and browser rendering, and is not a benchmark. Warm up both
models and use your evaluation notebook for fair speed comparisons.

### Adapter shape

```python
# adapters/yolo.py or adapters/faster_rcnn.py
# Import your own loader and inference routine, then load weights once.
# model = load_your_trained_model(...)

READY = True  # only after the implementation below is complete
STATUS = "Trained model connected"

def predict(image):
    # 1. Prepare the image as your trained detector expects.
    # 2. Run the selected detector, decode outputs and apply final NMS.
    # 3. Map boxes back to original image pixels.
    # 4. Return a list of detection dictionaries using the contract above.
    raise NotImplementedError("Replace with your model's inference routine")
```

The architecture and weights are not available yet, so these adapters deliberately
avoid guessing how to reconstruct either detector. When the model code is ready,
the website files should not need to change. A Colab-only model would need an
accessible inference service or to be moved to the machine running this server;
the current adapters are for Python models running with the app.

## API

- `GET /api/models` returns each model's ID, name, readiness and status message.
- `POST /api/predict` takes JSON `{"model": "yolo", "image": "<base64 file bytes>"}`
  (or `"faster_rcnn"`). Returns `model`, `detections`, `inference_ms`, `width`,
  `height` and `image` (base64 PNG after orientation/alpha handling).
- Input: JPG, PNG or WebP, up to 12 MB and 24 megapixels.
- An unconnected model returns HTTP 503 with an explanatory message. Invalid
  uploads return 400/413. Adapter failures return 500; details go to the terminal.
- Requests are serialized during inference so the models can share a device.

## Check the app

```sh
gui/.venv/bin/python -m unittest discover -s gui/tests -v
```

The tests exercise the HTTP boundary, invalid uploads, unavailable models,
the output contract and letterbox coordinate restoration. Test predictions are
isolated inside the test suite and are never exposed by the running app.

Manual checks: upload an image, select each model, check the unavailable-model
message, remove/replace the image, resize the browser and use the keyboard.
After connection: run both models on a training/validation image, verify box
alignment on a portrait image and toggle views. Keep final-test
images out of threshold tuning and model-development decisions.

## Record and submit

For the GUI recording, show image upload, a model selection and a completed
prediction with boxes, labels and confidence scores. Demonstrate both models
once they are connected. Append this recording after the three-minute
inference portion of `Demo_Video.mp4`; the GUI segment is additional time.

The briefing permits exactly four files inside the submission ZIP. This GUI
folder is development source: do not add it as an extra folder to the final ZIP.
If including the GUI implementation in the submitted code, incorporate the
relevant source into the permitted model notebooks and describe it in
`README.txt`. Keep `Demo_Video.mp4` and the required filenames as specified.
