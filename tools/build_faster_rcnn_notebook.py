"""Build FasterRCNN_Project.ipynb: the Faster R-CNN project as one self-contained, narrative notebook.

Same structure as YOLO_Project.ipynb (tools/build_notebook.py): the shared data pipeline and evaluator, then
Person 2's Faster R-CNN (models/faster_rcnn.py) and its training / validation / final-test engine
(engine/faster_rcnn.py), with an explanation before each part and a figure or table after it.

    python tools/build_faster_rcnn_notebook.py
"""

from notebook_parts import ROOT, check_and_write, code, md, read_modules

OUTPUT = ROOT / "FasterRCNN_Project.ipynb"
MODULES = ["config.py", "boxes.py", "data/dataset.py", "data/transforms.py", "data/augment.py", "data/sampling.py",
           "models/faster_rcnn.py", "evaluation/records.py", "evaluation/metrics.py", "evaluation/protocol.py",
           "evaluation/report.py", "evaluation/benchmark.py", "engine/trainer.py", "engine/workflow.py",
           "engine/faster_rcnn.py", "visualize.py"]
SOURCES, IMPORTS = read_modules(MODULES)
IMPORTS = [line for line in IMPORTS if "yolo" not in line.lower()]
USED = set()


def src(module, *sections, title=None):
    """A code cell with package source, headed by the file it comes from."""
    USED.update((module, name) for name in sections)
    body = "\n\n\n".join(SOURCES[module][name] for name in sections)
    return code(f"# workbench_detection/{module}" + (f" — {title}" if title else "") + "\n" + body)


CELLS = []
add = CELLS.append

add(md("""
# Workbench tool detection with Faster R-CNN / ResNet-18

**EE3703 group project, Group 05.** The detector finds **drills, scissors and screwdrivers** in 640 × 640
workbench photos and reports a box, a class and a confidence for each. It is a two-stage detector: a region
proposal network (RPN) suggests where objects may be, and an RoI head classifies and refines each proposal.

This notebook is the whole Faster R-CNN project in reading order. Data, evaluation and the experiment protocol are
the same as in `YOLO_Project.ipynb`, so the two detectors are compared on equal terms.

| § | Section | What you see |
|---|---|---|
| 1 | Setup | every setting |
| 2 | Data | split sizes, class balance, labelled samples |
| 3 | Preprocessing and augmentation | letterboxing, augmented samples, class-aware sampling |
| 4 | Model | ResNet-18 backbone, anchors, RPN, RoI head |
| 5 | Loss and decoding | anchor labels, the RPN and RoI losses, proposals and detections |
| 6 | Evaluation | COCO AP, P/R/F1, the validation protocol |
| 7 | Training | loop and optimizer; training curves |
| 8 | Validation | chosen RPN NMS and confidence, precision-recall, where objects are lost |
| 9 | Final test | the single evaluation on the 50 original photos, speed and size |
| 10 | Error analysis | which mistakes the detector makes, with examples |
| 11 | Inference | detections on photos |
| 12 | Results and conclusions | metrics against the proposal's targets |

### What was implemented by the group, and what comes from libraries

| Component | Implementation |
|---|---|
| Dataset loading, letterboxing, augmentation, class-aware sampling | Group (NumPy / OpenCV operations) |
| RPN and RoI head, anchor labelling, box encoding/decoding, RPN and RoI losses, proposal filtering, RoI sampling, class-wise detection | Group, with individual PyTorch layers |
| Training loop, optimizer settings | Group |
| Precision/recall/F1, error categories, NMS and confidence selection, timing | Group |
| ResNet-18 backbone with ImageNet weights, anchor generator, RoIAlign, NMS and IoU operators | torchvision |
| AP50 and AP50:95 | pycocotools (official COCO evaluator) |
| Layers, optimizer (SGD), autograd | PyTorch |

**Data protocol.** 358 training and 92 validation images select every setting. The 50 original test photos are used
once, at the end, after retraining on train + validation with the frozen settings.

**Running it.** On Colab: Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all, and upload
`dataset.zip` (from the repository root) when asked. Every setting is in the *Settings* cell of section 1. The final
test is off until `RUN_FINAL_TEST = True`.
"""))

# 1 Setup
add(md("## 1. Setup"))
add(code("""
# Install only what is missing (Colab already has PyTorch, torchvision, NumPy, Pillow, OpenCV and Matplotlib).
import importlib.util, subprocess, sys
missing = [package for module, package in [("pycocotools", "pycocotools"), ("cv2", "opencv-python-headless")]
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *missing], check=True)
"""))
add(code("# Imports used throughout the notebook\n" + "\n".join(sorted(IMPORTS, key=lambda s: (s.startswith("from"), s)))
         + "\nimport subprocess\nimport sys\nimport matplotlib\nimport matplotlib.pyplot as plt\nfrom datetime import datetime"))
add(md("""
Project constants and the training configuration. `FasterRCNNConfig` holds every setting that affects training; it is
saved with the run and inside each checkpoint, so the final retrain uses exactly what validation selected.
"""))
add(src("config.py", "constants"))
add(src("models/faster_rcnn.py", "settings", title="Faster R-CNN constants"))
add(src("engine/faster_rcnn.py", "config"))
add(md("""
### Settings
The only cell to edit. `TRIAL` picks the recipe:

* **`"baseline"`**: Person 2's development recipe (`FasterRCNNConfig()`), measured at 26.6% validation AP50 on
  Person 2's branch, where the RPN covered only 2 of 47 validation screwdrivers in its top 100 proposals.
* **`"accuracy"`**: the same architecture with changes aimed at that weakness: longer training (up to 60 epochs with
  early stopping after 10 without improvement, as in the proposal) with warmup and cosine decay at a higher learning rate; anchor ratios 0.25 and 4 for long, thin tools; RPN foreground from IoU 0.6
  instead of 0.7 (thin objects rarely reach 0.7 with any anchor); and 128 RoIs per image for the RoI head
  (up to 32 foreground) instead of 32 (up to 8), from 600 training proposals.

The recipe with the higher validation AP50 is kept for the final test.

Every stage saves its results in the run folder (Google Drive on Colab) and is skipped when re-run if it has already
finished. **After a disconnect**, set `RESUME_RUN` to the run folder's name (printed below) and Run all. **For the
final test**, once the validation results are good, set `RUN_FINAL_TEST = True` (with `RESUME_RUN` if the session has
restarted) and Run all again.
"""))
add(code("""
TRIALS = {
    "baseline": FasterRCNNConfig(),  # dense anchors on stride-16 features, SGD lr 5e-4 (heads 4x), batch 2, 10 epochs
    "accuracy": FasterRCNNConfig(epochs=60, lr=1e-3, lr_schedule="cosine", warmup_epochs=1.0,
                                 aspect_ratios=(0.25, 0.5, 1.0, 2.0, 4.0), rpn_positive_iou=0.6,
                                 train_proposals=600, roi_samples=128, roi_foreground=32),
}
TRIAL = "accuracy"
CONFIG = TRIALS[TRIAL]
RUN_FINAL_TEST = False       # the single evaluation on the 50 original test photos; leave False until settings are final
SAVE_TO_DRIVE = True         # on Colab, keep the run in Google Drive in case the session disconnects
DATASET = Path("/content/dataset") if Path("/content").is_dir() else Path("dataset")
RESUME_RUN = None            # e.g. "frcnn_accuracy_20261104_2130" to continue an earlier run
RUN_NAME = RESUME_RUN or f"frcnn_{TRIAL}_" + datetime.now().strftime("%Y%m%d_%H%M")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_COLAB = "google.colab" in sys.modules
if not (DATASET / "manifest.csv").is_file() and IN_COLAB:
    from google.colab import files
    archives = [path for path in (Path("/content/dataset.zip"), Path("/content/dataset_fixed.zip")) if path.is_file()]
    if not archives:
        print("Select the dataset zip (dataset.zip from the repository)")
        archives = [Path(name).resolve() for name in files.upload()]  # saved under the uploaded file's own name
    subprocess.run(["unzip", "-q", "-o", str(archives[0]), "-d", "/content"], check=True)
assert (DATASET / "manifest.csv").is_file(), f"Dataset not found at {DATASET}"
if SAVE_TO_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    RUNS = Path("/content/drive/MyDrive/EE3703_frcnn_runs")
else:
    RUNS = Path("runs")
seed_everything()
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
print(f"Dataset: {DATASET} | run folder: {RUNS / RUN_NAME}")
print(f"Trial: {TRIAL} | configuration:", CONFIG.to_dict())
"""))

# 2 Data
add(md("""
## 2. Data

500 images in three splits. **Originals** are the group's own workbench photos; **public** images come from open
datasets. Every image is already letterboxed to 640 × 640 and labelled in YOLO and COCO formats; `manifest.csv`
lists each image's split, origin, source and per-class box counts. The test split holds only originals and is
locked: loading it needs `allow_test=True`.
"""))
add(src("data/dataset.py", "dataset"))
add(md("Drawing helpers for the figures in this notebook (boxes in green are ground truth, coloured boxes are predictions)."))
add(src("visualize.py", "drawing", "data-figures", title="figures"))
add(code("""
rows = read_manifest(DATASET)
table = split_summary(rows)
print(f"{'split':<7}{'images':>8}{'original':>10}{'public':>8}" + "".join(f"{name:>13}" for name in CLASSES))
for row in table:
    print(f"{row['split']:<7}{row['images']:>8}{row['original']:>10}{row['public']:>8}" + "".join(f"{row[n]:>13}" for n in CLASSES))
plot_class_balance(table)
"""))
add(code("""
train_set = ToolDataset(DATASET, "train")
originals = [i for i, origin in enumerate(train_set.origins) if origin == "original"]
public = [i for i, origin in enumerate(train_set.origins) if origin == "public"]
show_samples(train_set, originals[:4] + public[:4])
"""))

# 3 Preprocessing and augmentation
add(md("""
## 3. Preprocessing and augmentation

The same pipeline as the YOLOv2 notebook. **Letterboxing** scales the long side to 640 and pads with grey, so shapes
are never distorted; new photos (section 11) go through the same function.
"""))
add(src("data/transforms.py", "letterbox"))
add(md("""
**Training augmentation** (training images only, a different draw every time an image is used): horizontal flip,
rotation within ±15°, shift and scale, brightness/contrast/gamma, hue/saturation, blur or sensor noise, a random grey
cutout, and for scissors a wider scale range and partial occlusion. Every box must stay at least 30% visible.
Faster R-CNN trains without mosaic (`CONFIG.mosaic = 0`).
"""))
add(src("data/augment.py", "settings", "geometric", "photometric", "occlusion", "pipelines", "mosaic"))
add(code("""
scissor_image = next(i for i, labels in enumerate(train_set.labels) if 1 in labels)
show_augmentations(ToolDataset(DATASET, "train", augment=True, augmentation=CONFIG.augmentation), scissor_image)
"""))
add(md("""
**Class-aware sampling.** Each training image is drawn with a weight that boosts the rarest class it contains
(inverse box frequency, capped at 2×), so every class appears about equally often per epoch.
"""))
add(src("data/sampling.py", "sampling"))
add(code("""
weights = image_weights(train_set, CONFIG.original_weight)
uniform, balanced = expected_box_counts(train_set, [1.0] * len(train_set)), expected_box_counts(train_set, weights)
print("Expected boxes per epoch:", {n: f"{u:.0f} -> {b:.0f}" for n, u, b in zip(CLASSES, uniform, balanced)})
plot_box_counts(uniform, balanced)
"""))

# 4 Model
add(md("""
## 4. Model

Box utilities used by the evaluator (IoU of two boxes and format conversion).
"""))
add(src("boxes.py", "box-ops"))
add(md("""
**Architecture.** A ResNet-18 backbone (ImageNet weights) produces a feature map. With the `dense` anchor profile it
stops after layer 3 (stride 16, 256 channels, a 40 × 40 map) so small screwdrivers cover several feature cells; the
`original` profile uses layer 4 (stride 32, 512 channels).

* **Anchors**: at every feature cell, one box per size × aspect ratio (dense: 5 sizes 32–512 px × ratios 0.5/1/2 = 15;
  the accuracy trial adds ratios 0.25 and 4, giving 25).
* **RPN**: a 3×3 convolution, then per anchor one objectness score and four box offsets.
* **RoI head**: RoIAlign pools each proposal to 5 × 5 (dense) or 7 × 7 features; a fully connected layer of 1024
  units feeds a 4-way classifier (background + 3 tools) and class-specific box offsets.
"""))
add(src("models/faster_rcnn.py", "model"))
add(code("""
model = FasterRCNN_Modular(len(CLASSES) + 1, pretrained=CONFIG.pretrained_backbone,
                           anchor_scales=ANCHOR_PROFILES[CONFIG.anchor_profile], aspect_ratios=CONFIG.aspect_ratios)
parts = {"backbone (ResNet-18)": model.backbone, "RPN": model.rpn, "RoI head": model.stage2}
for name, module in parts.items():
    print(f"{name}: {sum(p.numel() for p in module.parameters()):,} parameters")
image, target = train_set[scissor_image]
with torch.no_grad():
    features, anchors, rpn_cls, rpn_reg = model.eval()([image])
print(f"Feature map {tuple(features.shape)} | {len(anchors[0]):,} anchors | RPN scores {tuple(rpn_cls.shape)} "
      f"| RPN offsets {tuple(rpn_reg.shape)}")
"""))

# 5 Loss and decoding
add(md("""
## 5. Loss and decoding

**RPN loss.** An anchor is foreground if it overlaps a ground-truth box above `CONFIG.rpn_positive_iou` (0.7, or 0.6 in
the accuracy trial), background below 0.3, and ignored
in between; each object's best-matching anchor is always foreground. Objectness uses binary cross-entropy and box
offsets smooth L1 (positives only). With the dense profile, 256 anchors are sampled per image (at most half
foreground) to balance the two.

**RoI loss.** Proposals overlapping an object at IoU > 0.5 take its class, the rest are background. Classification
uses cross-entropy (balanced between foreground and background with the dense profile) and box offsets smooth L1 for
the true class. The total loss is RPN + RoI, with box terms weighted by λ = 10.
"""))
add(src("models/faster_rcnn.py", "loss"))
add(code("""
iou, _ = _match_boxes_to_gt(anchors[0], target["boxes"])
labels, _, _, _ = _create_rpn_labels(iou, iou.device, anchors[0], target["boxes"], CONFIG.rpn_positive_iou)
print(f"{target['file_name']}: {len(target['boxes'])} objects -> {int((labels == 1).sum())} foreground, "
      f"{int((labels == 0).sum()):,} background and {int((labels == -1).sum()):,} ignored anchors")
"""))
add(md("""
**Decoding.** RPN offsets turn anchors into proposals; the 10 × limit highest-scoring go through NMS at the RPN NMS
IoU (0.70 during training; 0.50 or 0.70 chosen on validation), and the best 600 (dense) are kept. The RoI head then
scores each proposal per class; scores are multiplied by √(RPN objectness) with the dense profile, and class-wise NMS
at IoU 0.50 removes duplicates.
"""))
add(src("models/faster_rcnn.py", "decoding"))
add(code("""
with torch.no_grad():
    proposals, objectness = filter_proposals(rpn_cls, rpn_reg, anchors[0], image.shape[-2:], limit=10)
_, axis = plt.subplots(figsize=(5, 5))
draw_boxes(axis, denormalize(image), proposals.tolist(), [0] * len(proposals), colour="#ff7f0e",
           title="Top 10 proposals of the untrained RPN (class names not meaningful)")
plt.show()
"""))

# 6 Evaluation
add(md("""
## 6. Evaluation

The same evaluator as the YOLOv2 notebook. Predictions and ground truth are checked strictly (every image needs a
record; boxes inside the canvas; scores in [0, 1]) so nothing is silently dropped.
"""))
add(src("evaluation/records.py", "helpers", "validation", "ground-truth", "export"))
add(md("""
**Metrics.** AP50 and AP50:95 come from the official COCO evaluator (pycocotools). Precision, recall, F1 and mean
matched IoU are computed at an *operating confidence* chosen on validation. Every false positive is labelled as a
duplicate, wrong class, poor localization or background detection, and every unmatched object as missed.
"""))
add(src("evaluation/metrics.py", "config", "matching", "coco", "evaluate"))
add(md("""
**Protocol.** On validation, the detector exports predictions at each RPN NMS IoU in the proposal's grid for
Faster R-CNN (0.50 / 0.70, with the final class-wise NMS fixed at 0.50); the best AP50 wins, and the operating
confidence (0.2 / 0.3 / 0.4) is the one with the best F1. These are frozen in `evaluation_protocol.json`, which is
the only way to score the test split.
"""))
add(src("evaluation/protocol.py", "grids", "selection", "run"))
add(src("evaluation/report.py", "svg", "summary", title="saved figures and printed summaries"))
add(src("evaluation/benchmark.py", "benchmark", title="inference speed and model size"))

# 7 Training
add(md("""
## 7. Training

**Inference** goes through one detector object, used for validation during training, the final test, the error
analysis and the demo, so they all share the same preprocessing, decoding and thresholds.
"""))
add(src("engine/faster_rcnn.py", "detector"))
add(md("""
**The training loop** (Person 2's recipe). Each image goes through the network on its own: the RPN proposes
`train_proposals` boxes, the ground-truth boxes are added, and `roi_samples` RoIs are sampled (at most `roi_foreground`
foreground) for the RoI head. SGD with momentum 0.9; with the dense profile the RPN and RoI head learn 4× faster than
the backbone, the backbone's BatchNorm statistics stay fixed (one image per forward pass is too few to estimate them),
and gradients are clipped at norm 5. The learning rate is constant (baseline) or warms up and then follows a cosine
decay to 5% (accuracy trial; the same schedule as YOLOv2). Validation AP50 is measured every epoch; the best epoch is
kept.
"""))
add(src("engine/trainer.py", "loader", "schedule", title="data loaders and learning-rate schedule (shared with YOLOv2)"))
add(src("engine/faster_rcnn.py", "train"))
add(md("**The workflow** ties the stages together and keeps every run's files in one folder, as for YOLOv2."))
add(src("engine/workflow.py", "run"))
add(src("engine/faster_rcnn.py", "workflow"))
add(src("visualize.py", "result-figures", title="figures for training and results"))
add(md("### Software check\nOne update and one evaluation on 8 training images, to confirm that boxes, class IDs, model and evaluator "
       "connect before the long run. Its scores are not results."))
add(code("""
RUN = FasterRCNNRun(RUNS / RUN_NAME)
if RUN.trained_checkpoint() is None:
    frcnn_software_check(DATASET, RUN.root / "software_check", DEVICE)
else:
    print("Training already finished in this run; software check skipped.")
"""))
add(md("### Development training\nTrains on the 358 training images and validates on the 92 validation images every epoch."))
add(code("""
checkpoint = RUN.trained_checkpoint()
if checkpoint is None:
    checkpoint = frcnn_develop(CONFIG, DATASET, RUN, DEVICE)
else:
    CONFIG = FasterRCNNConfig.from_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["config"])
    print(f"Loaded finished training: {checkpoint}\\nSettings: {CONFIG.to_dict()}")
with (Path(checkpoint).parent / "history.csv").open(encoding="utf-8") as handle:
    history = [{key: float(value) for key, value in row.items() if value != ""} for row in csv.DictReader(handle)]
plot_history(history, "Development")
"""))

# 8 Validation
add(md("""
## 8. Validation

Settings are chosen here and frozen for the test: the RPN NMS IoU with the best AP50 and the operating confidence
with the best F1. Original and public validation images are reported separately to expose any domain gap.
"""))
add(code("""
validation = RUN.result("validation")
if validation is None:
    validation = frcnn_validate(checkpoint, DATASET, RUN, DEVICE)
else:
    print("Loaded finished validation:\\n" + (RUN.report("validation") / "metrics.txt").read_text())
protocol = load_json(RUN.protocol_path)
settings = protocol["models"][FRCNN_NAME]
print("RPN NMS trials:", {row["nms_iou"]: round(row["map50"], 4) for row in protocol["nms_trials"][FRCNN_NAME]})
print("Confidence trials (F1):", {row["confidence"]: round(row["micro_f1"], 4) for row in settings["confidence_trials"]})
print(f"Frozen settings: RPN NMS IoU {settings['nms']['nms_iou']}, final NMS IoU {settings['nms']['final_nms_iou']}, "
      f"operating confidence {settings['operating_confidence']}")
plot_pr_curves(validation)
"""))
add(md("""
**Where objects are lost** (Person 2's diagnostics). For each validation object: does any RPN proposal cover it at
IoU ≥ 0.5 (all proposals / top 100), and does the RoI head name the right class when given the exact ground-truth
box? A class with low coverage is lost in the RPN; one with good coverage but low AP is lost in the RoI head.
"""))
add(code("""
detector = FasterRCNNDetector.from_checkpoint(checkpoint, DEVICE, protocol)
diagnostics = pipeline_diagnostics(detector, ToolDataset(DATASET, "valid"))
print(f"{'class':<13}{'objects':>8}{'RPN covered':>13}{'top 100':>9}{'RoI correct on exact box':>26}")
for name, row in diagnostics.items():
    print(f"{name:<13}{row['objects']:>8}{row['rpn_covered']:>13}{row['rpn_top100_covered']:>9}"
          f"{row['roi_correct_on_exact_boxes']:>26}")
"""))
add(code("""
valid_truth = load_ground_truth(DATASET, "valid")
examples = pick_examples(validation)
shown = examples["success"][:2] + examples["missed_object"] + examples["background_or_low_overlap"]
if shown:
    show_predictions(detector, DATASET, "valid", valid_truth, shown)
"""))

# 9 Final test
add(md("""
## 9. Final test

Retrains from ImageNet on train + validation (450 images) for the development run's best epoch count, with every
setting frozen, then evaluates **once** on the 50 original test photos and measures inference speed (batch 1,
including preprocessing, decoding and NMS).
"""))
add(code("""
test = RUN.result("test")
if test is None and RUN_FINAL_TEST:
    test = frcnn_final_test(RUN, DATASET, DEVICE)
elif test is not None:
    print("Loaded the finished test (it is never repeated):\\n" + (RUN.report("test") / "metrics.txt").read_text())
if test is not None:
    timing = test["model_metadata"]["benchmark"]
    size = test["model_metadata"]["model_size"]
    print(f"Speed: {timing['mean_latency_ms']:.1f} ms/image ({timing['sequential_fps']:.1f} FPS) on {timing['device']}")
    print(f"Model: {size['parameter_count']:,} parameters, checkpoint {size['checkpoint_file_bytes'] / 1e6:.1f} MB")
    plot_pr_curves(test)
else:
    print("Final test not run. Set RUN_FINAL_TEST = True once every setting is fixed on validation.")
"""))

# 10 Error analysis
add(md("""
## 10. Error analysis

Error types at the operating confidence, on the test split if it has been run, otherwise on validation. Categories
describe each mistake; they are not proven causes.
"""))
add(code("""
analysed, analysed_split = (test, "test") if test else (validation, "valid")
counts = plot_error_breakdown(analysed)
if test:
    detector = FasterRCNNDetector.from_checkpoint(RUN.final_checkpoint(), DEVICE, protocol)
truth = load_ground_truth(DATASET, analysed_split, allow_test=analysed_split == "test")
examples = pick_examples(analysed)
for kind, ids in examples.items():
    if ids and kind != "success":
        print(kind.replace("_", " "))
        show_predictions(detector, DATASET, analysed_split, truth, ids)
"""))

# 11 Inference
add(md("""
## 11. Inference on photos

`predict_image` takes a photo of any size, letterboxes it like the training data, detects and maps the boxes back to
the photo's pixels. This is the same call the browser demo uses.
"""))
add(code("""
split = "test" if test else "valid"
folder = DATASET / split / "images"
photos = sorted(folder.iterdir())[:6]
_, axes = plt.subplots(2, 3, figsize=(13, 8.5))
for axis, path in zip(axes.flat, photos):
    image = Image.open(path).convert("RGB")
    found = detector.predict_image(image)
    axis.imshow(draw_detections(image, found))
    axis.set_title(f"{path.name}: {len(found)} detections", fontsize=9)
    axis.axis("off")
plt.tight_layout()
plt.show()
if test is not None:  # every test photo with its detections, for the demo video
    saved = RUN.root / "test_predictions"
    saved.mkdir(exist_ok=True)
    for path in sorted(folder.iterdir()):
        image = Image.open(path).convert("RGB")
        draw_detections(image, detector.predict_image(image)).save(saved / path.name, quality=92)
    print(f"Saved {len(list(saved.iterdir()))} annotated test photos to {saved}")
# Your own photo: detector.predict_image(Image.open("photo.jpg").convert("RGB"))
"""))

# 12 Results
add(md("## 12. Results and conclusions\n\nAchieved AP50 against the targets in the proposal (sections 1.3 and 1.4)."))
add(code("""
final = test or validation
targets = {"overall": 0.70, "drill": 0.75, "scissor": 0.65, "screwdriver": 0.70}
f3 = lambda value, width: f"{'n/a':>{width}}" if value is None else f"{value:>{width}.3f}"
print(f"{'split: ' + ('test (50 originals)' if test else 'validation (not the final result)'):<40}")
print(f"{'':<14}{'AP50':>8}{'target':>9}{'AP50:95':>10}{'precision':>11}{'recall':>8}{'F1':>7}")
print(f"{'overall':<14}{f3(final['map50'], 8)}{targets['overall']:>9.2f}{f3(final['map_iou_average'], 10)}"
      f"{f3(final['precision_micro'], 11)}{f3(final['recall_micro'], 8)}{f3(final['f1_micro'], 7)}")
for row in final["per_class"]:
    print(f"{row['class_name']:<14}{f3(row['ap50'], 8)}{targets[row['class_name']]:>9.2f}{f3(row['ap_iou_average'], 10)}"
          f"{f3(row['precision'], 11)}{f3(row['recall'], 8)}{f3(row['f1'], 7)}")
for name, subset in final["subsets"].items():
    print(f"{name:<22} AP50 {f3(subset['map50'], 0).strip()} ({subset['image_count']} images)")
print(f"\\nRun folder: {RUN.root.resolve()}")
"""))
add(md("""
### Conclusions

*To be written from the results above:* how close each class came to its target; whether objects are lost in the RPN
or the RoI head (section 8); what the error analysis shows; how Faster R-CNN compares with YOLOv2 on accuracy and
speed; and what would improve it next.
"""))

check_and_write(OUTPUT, CELLS, SOURCES, USED, required=["models/faster_rcnn.py", "engine/faster_rcnn.py"])
