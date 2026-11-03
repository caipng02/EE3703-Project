# Workbench Tool Detection

EE3703 group project, Group 05. Detects **drills, scissors and screwdrivers** in workbench photos and reports
a bounding box, class label and confidence for each. The project compares a two-stage **Faster R-CNN** with a
one-stage **YOLOv2**, both on a ResNet-18 backbone, for inventory and safety assistance.

**Start here:** [`YOLO_Project.ipynb`](YOLO_Project.ipynb) walks through the YOLOv2 system end to end (data,
augmentation, model, loss, training, validation, final test, error analysis and inference) and is the
submitted YOLO notebook.

## System overview

```
dataset/ ──► data ──────────► models ─────────► engine.trainer ──► engine.predictor ──► evaluation
             letterbox,        YOLOv2/ResNet-18,  AdamW, cosine,     one detector for    COCO AP, P/R/F1,
             augmentation,     anchors, loss,     EMA, search,       validation, test,   NMS/confidence
             class-aware       decoding, NMS      checkpoints        GUI and demo        protocol, errors
             sampling
```

| Package | Contents |
|---|---|
| `workbench_detection/config.py` | Constants and `TrainConfig`, every training setting in one place |
| `workbench_detection/boxes.py` | Box formats, IoU and NMS, shared by the model and evaluation |
| `workbench_detection/data/` | Dataset, letterboxing, augmentation and mosaic, class-aware sampling |
| `workbench_detection/models/` | YOLOv2 / ResNet-18, ImageNet initialization, anchors, decoding, loss |
| `workbench_detection/engine/` | Training loop, hyperparameter search, predictor, experiment workflow |
| `workbench_detection/evaluation/` | Ground-truth loading, metrics, validation protocol, reports, timing |
| `workbench_detection/gui/` | Browser demo (upload, model choice, boxes with labels and confidences) |
| `workbench_detection/visualize.py` | Figures used by the notebook |
| `configs/yolov2.yaml` | The recommended training settings |
| `tools/build_notebook.py` | Regenerates `YOLO_Project.ipynb` from the package |
| `tests/` | GUI and upload checks |
| `experiments/` | Person 3's earlier standalone YOLOv2 experiments (round 2.2) |

### Implementation boundary

The YOLOv2 detection pipeline is written by the group from individual PyTorch layers: the ResNet-18 blocks,
detection head and passthrough layer, k-means anchors, anchor assignment, box encoding and decoding, the loss,
confidence filtering and class-wise NMS. So are the data pipeline, training loop, search and the evaluation
logic. Library code: ImageNet weights for the backbone (torchvision, allowed by the professor), AP50 and AP50:95
(pycocotools), and PyTorch's layers, optimizer and autograd.

### Data protocol

358 training and 92 validation images select every setting. The test split (50 original photos) is locked in
code: the final command retrains on train + validation with the frozen settings and evaluates it once.

## Setup

Python 3.10 or newer. Install [PyTorch](https://pytorch.org/get-started/locally/) for your CPU or GPU, then:

```sh
python -m pip install -r requirements.txt
```

Unpack `dataset.zip` (in the repository root; built by Person 1's pipeline on the data branch) into `dataset/`
(git-ignored), e.g. `unzip dataset.zip`. On Colab, upload it to the notebook when asked. On Intel Macs PyTorch stops at 2.2, which needs
`python -m pip install "numpy<2" "opencv-python-headless<4.11"`.

## Usage

Everything runs through one command; each run keeps its settings, checkpoints and reports in `runs/<name>/`.

```sh
python -m workbench_detection inspect                          # 1-minute software check
python -m workbench_detection train --run runs/v1              # develop + validate with configs/yolov2.yaml
python -m workbench_detection train --run runs/v2 --set lr=3e-4 ema=false   # override settings
python -m workbench_detection train --run runs/search --search 8            # the proposal's random search
python -m workbench_detection test --run runs/v1               # final retrain + the single test evaluation
python -m workbench_detection predict --run runs/v1 photo.jpg  # draw detections on your own photos
python -m workbench_detection gui --run runs/v1                # browser demo at http://127.0.0.1:8765
python -m workbench_detection score --help                     # score another detector's prediction JSON
```

A run folder:

```
runs/v1/config.yaml                                 settings used
runs/v1/work/train/best.pt, history.csv
runs/v1/work/validation/evaluation_protocol.json    NMS IoU and confidence chosen on validation
runs/v1/report/validation/metrics.txt, examples/    metrics, figures and example predictions
runs/v1/work/final_train/, report/test/             after the test command
```

Training on Colab: open `YOLO_Project.ipynb`, choose a T4 GPU, Run all, and upload `dataset_fixed.zip` when asked.
After changing the package, rebuild the notebook with `python tools/build_notebook.py`.

### Adding Faster R-CNN

Faster R-CNN plugs into the same data pipeline, evaluation protocol and GUI: export predictions in the
evaluator's format (pixel xyxy boxes on the 640 canvas, class IDs 0–2, scores, `model_family: faster_rcnn`),
score them with `score --project-protocol`, and implement `predict(image)` in
`workbench_detection/gui/adapters/faster_rcnn.py` like the YOLOv2 adapter.

## Team

| Role | Responsibility |
|---|---|
| Person 1 | Dataset strategy, annotation, preprocessing and data limitations |
| Person 2 | Faster R-CNN methodology, implementation and results |
| Person 3 | YOLOv2 methodology, implementation and results |
| Person 4 | Experimental design, evaluation metrics, comparisons, error analysis and final consolidation |
| Persons 1 and 4 | Interactive demo GUI |

## Deliverables

| Deliverable | Deadline | Weight |
|---|---|---|
| Project proposal (`Proposal_Project_Group_05.pdf`) | 6 October 2026, week 8 | 10% |
| Project report (`Report_Project_Group_05.pdf`) | 6 November 2026, week 12 | 10% |
| Demo video and code (`Demo_Project_Group_05.zip`) | 6 November 2026, week 12 | 10%, plus up to 2% GUI bonus |
| In-class presentation (12 min + 5 min Q&A) | Week 13, Tuesday class | 10% |

The demo zip must contain **exactly four files**: `Demo_Video.mp4` (3-minute inference on the 50 original test
images by both detectors, with the GUI recording appended after it), `FasterRCNN_Project.ipynb`,
`YOLO_Project.ipynb` and `README.txt`. Notebooks must be run before submission with their outputs saved; the
teaching team inspects outputs rather than running the code. The report must show successful and failed
detections and state which components were implemented manually and which come from libraries.
