"""Build YOLO_Project.ipynb: the project as one self-contained, narrative notebook.

The notebook inlines the workbench_detection source in reading order (data -> model -> loss ->
evaluation -> training -> results), with an explanation before each part and a figure or table
after it. Code comes from the package itself, so the notebook and the package cannot drift apart.

Each module marks its sections with "# %% <name>" comments; the builder places those sections,
drops the package-relative imports (every name lives in the notebook's one namespace) and gathers
the third-party imports into the setup cell.

    python tools/build_notebook.py
"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "workbench_detection"
OUTPUT = ROOT / "YOLO_Project.ipynb"
RELATIVE_IMPORT = re.compile(r"^\s*from \.+[\w.]* import [^(]*$")


# --- reading the package -------------------------------------------------------------------------

def read_module(relative):
    """Return (import lines, {section name: code}) for one module."""
    text = (PACKAGE / relative).read_text(encoding="utf-8")
    parts = re.split(r"^# %% (\S+)\n", text, flags=re.M)
    header, sections = parts[0], dict(zip(parts[1::2], parts[2::2]))
    tree = ast.parse(header)
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                continue
            imports.append(ast.get_source_segment(header, node))
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            raise ValueError(f"{relative}: move code above the first '# %%' section into a section")
    cleaned = {}
    for name, code in sections.items():
        lines = [line for line in code.splitlines() if not RELATIVE_IMPORT.match(line)]
        cleaned[name] = "\n".join(lines).strip("\n")
    return imports, cleaned


MODULES = ["config.py", "boxes.py", "data/dataset.py", "data/transforms.py", "data/augment.py", "data/sampling.py",
           "models/yolov2.py", "models/loss.py", "evaluation/records.py", "evaluation/metrics.py",
           "evaluation/protocol.py", "evaluation/report.py", "evaluation/benchmark.py", "engine/predictor.py",
           "engine/trainer.py", "engine/search.py", "engine/workflow.py", "visualize.py"]
SOURCES = {}
IMPORTS = []
for module in MODULES:
    module_imports, sections = read_module(module)
    IMPORTS += [line for line in module_imports if line not in IMPORTS]
    SOURCES[module] = sections
USED = set()


def source(module, section):
    USED.add((module, section))
    return SOURCES[module][section]


# --- notebook cells ------------------------------------------------------------------------------

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def src(module, *sections, title=None):
    """A code cell with package source, headed by the file it comes from."""
    body = "\n\n\n".join(source(module, name) for name in sections)
    return code(f"# workbench_detection/{module}" + (f" — {title}" if title else "") + "\n" + body)


CELLS = []
add = CELLS.append

add(md("""
# Workbench tool detection with YOLOv2 / ResNet-18

**EE3703 group project, Group 05.** The detector finds **drills, scissors and screwdrivers** in 640 × 640
workbench photos and reports a box, a class and a confidence for each.

This notebook is the whole YOLOv2 project in reading order:

| § | Section | What you see |
|---|---|---|
| 1 | Setup | configuration of every setting |
| 2 | Data | split sizes, class balance, labelled samples |
| 3 | Preprocessing and augmentation | letterboxing, augmented samples, class-aware sampling |
| 4 | Model | ResNet-18 + YOLOv2 head, ImageNet initialization, anchors |
| 5 | Loss and decoding | anchor assignment and the four loss terms |
| 6 | Evaluation | COCO AP, P/R/F1, the validation protocol |
| 7 | Training | loop, schedule, weight averaging, search; training curves |
| 8 | Validation | chosen NMS and confidence, precision-recall, examples |
| 9 | Final test | the single evaluation on the 50 original photos, speed and size |
| 10 | Error analysis | which mistakes the detector makes, with examples |
| 11 | Inference | detections on photos |
| 12 | Results and conclusions | metrics against the proposal's targets |

### What was implemented by the group, and what comes from libraries

| Component | Implementation |
|---|---|
| Dataset loading, letterboxing, augmentation, mosaic, class-aware sampling | Group (NumPy / OpenCV operations) |
| ResNet-18 layers, YOLOv2 head and passthrough, k-means anchors, anchor assignment, box encoding/decoding, YOLOv2 loss, confidence filtering, class-wise NMS | Group, from scratch with individual PyTorch layers |
| Training loop, learning-rate schedule, weight averaging (EMA), hyperparameter search | Group |
| Precision/recall/F1, error categories, NMS and confidence selection, timing | Group |
| Backbone initialization | ImageNet ResNet-18 weights from torchvision, copied into the hand-built layers (allowed by the professor) |
| AP50 and AP50:95 | pycocotools (official COCO evaluator) |
| Layers, optimizer (AdamW), autograd | PyTorch |

**Data protocol.** 358 training and 92 validation images select every setting. The 50 original test photos are used
once, at the end, after retraining on train + validation with the frozen settings.

**Running it.** On Colab: Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all, and upload
`dataset.zip` (from the repository root) when asked. Every setting is in the *Settings* cell of section 1. The final test is off until
`RUN_FINAL_TEST = True`.
"""))

# 1 Setup
add(md("## 1. Setup"))
add(code("""
# Install only what is missing (Colab already has PyTorch, torchvision, NumPy, Pillow, OpenCV and Matplotlib).
import importlib.util, subprocess, sys
missing = [package for module, package in [("pycocotools", "pycocotools"), ("cv2", "opencv-python-headless"),
                                            ("yaml", "pyyaml")] if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *missing], check=True)
"""))
add(code("# Imports used throughout the notebook\n" + "\n".join(sorted(IMPORTS, key=lambda s: (s.startswith("from"), s)))
         + "\nimport subprocess\nimport sys\nimport matplotlib\nimport matplotlib.pyplot as plt\nfrom datetime import datetime"))
add(md("""
Project constants and the training configuration. `TrainConfig` holds every setting that affects training; it is
saved with the run and inside each checkpoint, so the final retrain uses exactly what validation selected.
"""))
add(src("config.py", "constants", "train-config"))
add(md("""
### Settings
The only cell to edit. `TrainConfig()` is the recommended recipe; change fields like `TrainConfig(lr=3e-4)`.

Every stage saves its results in the run folder (Google Drive on Colab) and is skipped when re-run if it has already
finished. **After a disconnect**, set `RESUME_RUN` to the run folder's name (printed below) and Run all: finished
training, validation and test are loaded instead of repeated. **For the final test**, once the validation results
are good, set `RUN_FINAL_TEST = True` (with `RESUME_RUN` if the session has restarted) and Run all again.
"""))
add(code("""
CONFIG = TrainConfig()     # pretrained backbone, full augmentation + mosaic 0.5, batch 8, lr 1e-3, cosine, EMA
SEARCH_CONFIGS = 0         # proposal section 4.2: 8 trains eight lr x weight decay x batch combinations
RUN_FINAL_TEST = False     # the single evaluation on the 50 original test photos; leave False until settings are final
SAVE_TO_DRIVE = True       # on Colab, keep the run in Google Drive in case the session disconnects
DATASET = Path("/content/dataset") if Path("/content").is_dir() else Path("dataset")
RESUME_RUN = None          # e.g. "yolov2_20261104_2130" to continue an earlier run
RUN_NAME = RESUME_RUN or "yolov2_" + datetime.now().strftime("%Y%m%d_%H%M")

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
    RUNS = Path("/content/drive/MyDrive/EE3703_yolov2_runs")
else:
    RUNS = Path("runs")
seed_everything()
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
print(f"Dataset: {DATASET} | run folder: {RUNS / RUN_NAME}")
print("Configuration:", CONFIG.to_dict())
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

**Letterboxing** scales the long side to 640 and pads with grey, so shapes are never distorted. The dataset was built
with exactly this rule; new photos (section 11) go through the same function.
"""))
add(src("data/transforms.py", "letterbox"))
add(md("""
**Training augmentation** (training images only, a different draw every time an image is used): horizontal flip,
rotation within ±15°, shift and scale, brightness/contrast/gamma, hue/saturation, blur or sensor noise, a random grey
cutout, and for scissors a wider scale range and partial occlusion of the blades or handles. Every box must stay at
least 30% visible; otherwise the change is retried or skipped. **Mosaic** joins four training images into one.
"""))
add(src("data/augment.py", "settings", "geometric", "photometric", "occlusion", "pipelines", "mosaic"))
add(code("""
scissor_image = next(i for i, labels in enumerate(train_set.labels) if 1 in labels)
show_augmentations(ToolDataset(DATASET, "train", augment=True, augmentation=CONFIG.augmentation), scissor_image)
show_augmentations(ToolDataset(DATASET, "train", augment=True, mosaic_p=1.0, augmentation=CONFIG.augmentation), 0, count=4)
"""))
add(md("""
**Class-aware sampling.** Each training image is drawn with a weight that boosts the rarest class it contains
(inverse box frequency, capped at 2×), so every class appears about equally often per epoch. `original_weight`
can additionally favour the group's own photos if they score lower on validation.
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

Box utilities first: format conversion, IoU and non-maximum suppression, used by the model, the loss and the evaluator.
"""))
add(src("boxes.py", "box-ops", "nms"))
add(md("""
**Architecture.** A ResNet-18 backbone (7×7 stem, four stages of basic residual blocks) produces stride-16 and
stride-32 features. The YOLOv2 head adds two 3×3 convolutions on the deep features, reorganizes the stride-16
features onto the 20 × 20 grid (passthrough layer) and concatenates both, then predicts for each of the 5 anchors in
every cell: x, y, w, h offsets, objectness and 3 class scores (5 × 8 = 40 output channels).
"""))
add(src("models/yolov2.py", "backbone", "detector"))
add(md("""
**Initialization.** The backbone starts from ImageNet weights, copied by name into the hand-built layers; the YOLOv2
layers start random. The last layer's objectness bias starts at 1% instead of 50%: with about 1.5 objects but 2000
anchors per image, a 50% start makes background errors dominate the first updates (measured below).
"""))
add(src("models/yolov2.py", "initialization"))
add(md("""
**Anchors** are the typical box shapes, found by k-means on the training boxes with 1 − IoU as the distance
(YOLOv2's dimension clusters). Each prediction is an offset from one of them.
"""))
add(src("models/yolov2.py", "anchors"))
add(src("visualize.py", "model-figures", title="figures"))
add(code("""
anchors = anchors_from_dataset(train_set)
print("Anchors (width, height as fractions of the image):", [tuple(round(float(v), 3) for v in a) for a in anchors])
plot_anchors(train_set, anchors)
model = YOLOv2ResNet18(anchors)
parts = {"backbone (stem, s1-s4)": [model.stem, model.s1, model.s2, model.s3, model.s4],
         "YOLOv2 layers (deep, passthrough, head)": [model.deep, model.passthrough, model.head]}
for name, modules in parts.items():
    print(f"{name}: {sum(p.numel() for m in modules for p in m.parameters()):,} parameters")
with torch.no_grad():
    print("Output shape for one image:", tuple(model(train_set[0][0][None]).shape), "= (batch, anchors, grid y, grid x, 5 + classes)")
"""))

# 5 Loss and decoding
add(md("""
## 5. Loss and decoding

**Decoding** turns offsets into boxes: centre = cell index + sigmoid offset; size = anchor × exp(offset). Scores are
objectness × class probability; detections above a confidence threshold go through class-wise NMS.
"""))
add(src("models/yolov2.py", "decoding"))
add(md("""
**Loss.** Each object is assigned to the grid cell containing its centre and to the best-matching free anchor there.
Four terms: location (box offsets), object (objectness of assigned anchors, target = IoU of their predicted box),
background (objectness of all other anchors) and class (cross-entropy).
"""))
add(src("models/loss.py", "loss"))
add(code("""
show_assignment(train_set, scissor_image, anchors)
images, batch_targets = collate([train_set[i] for i in range(8)])
for prior in (False, True):
    seed_everything()
    untrained = YOLOv2ResNet18(anchors)
    if prior:
        init_objectness_prior(untrained)
    with torch.no_grad():
        _, terms = yolo_loss(untrained(images), encode_targets(batch_targets), untrained.anchors)
    share = terms["background"] / terms["loss"]
    print(f"objectness prior {'on ' if prior else 'off'}: loss {terms['loss']:.1f} | background term {share:.0%} | "
          f"{terms['assigned']} assigned anchors out of {8 * 5 * 20 * 20}")
"""))

# 6 Evaluation
add(md("""
## 6. Evaluation

Predictions and ground truth are checked strictly (every image needs a record; boxes inside the canvas; scores in
[0, 1]) so nothing is silently dropped. The ground-truth loader cross-checks the COCO and YOLO labels of every image.
"""))
add(src("evaluation/records.py", "helpers", "validation", "ground-truth", "export"))
add(md("""
**Metrics.** AP50 and AP50:95 come from the official COCO evaluator (pycocotools). Precision, recall, F1 and mean
matched IoU are computed at an *operating confidence* chosen on validation. Every false positive is labelled as a
duplicate, wrong class, poor localization or background detection, and every unmatched object as missed.
"""))
add(src("evaluation/metrics.py", "config", "matching", "coco", "evaluate"))
add(md("""
**Protocol.** On validation, the detector exports predictions at each NMS IoU in the proposal's grid
(0.45 / 0.50 / 0.60); the best AP50 wins, and the operating confidence (0.2 / 0.3 / 0.4) is the one with the best F1.
These are frozen in `evaluation_protocol.json`, which is the only way to score the test split.
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
add(src("engine/predictor.py", "predictor"))
add(md("""
**The training loop.** AdamW with mixed precision on the GPU; a 3-epoch linear warmup, then cosine decay to 5% of the
learning rate; gradient clipping at norm 10. Validation AP50 is measured every epoch on an exponential moving
average (EMA) of the weights, which smooths the epoch-to-epoch noise; the best epoch is kept, and training stops
after 10 epochs without improvement (proposal section 4.2).
"""))
add(src("engine/trainer.py", "loader", "optimization", "train"))
add(md("**Hyperparameter search** (proposal section 4.2): random combinations of learning rate {1e-4, 3e-4, 1e-3}, "
       "weight decay {1e-4, 5e-4} and batch size {4, 8}, selected by validation AP50."))
add(src("engine/search.py", "search"))
add(md("**The workflow** ties the stages together and keeps every run's files in one folder."))
add(src("engine/workflow.py", "run", "stages"))
add(src("visualize.py", "result-figures", title="figures for training and results"))
add(md("### Software check\nOne update and one evaluation on 8 training images, to confirm that boxes, class IDs, model and evaluator "
       "connect before the long run. Its scores are not results."))
add(code("""
RUN = Run(RUNS / RUN_NAME)
if RUN.trained_checkpoint() is None:
    software_check(DATASET, RUN.root / "software_check", DEVICE)
else:
    print("Training already finished in this run; software check skipped.")
"""))
add(md("### Development training\nTrains on the 358 training images and validates on the 92 validation images every epoch."))
add(code("""
checkpoint = RUN.trained_checkpoint()
if checkpoint is None:
    checkpoint = develop(CONFIG, DATASET, RUN, DEVICE, search=SEARCH_CONFIGS)
else:
    CONFIG = TrainConfig.from_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["config"])
    print(f"Loaded finished training: {checkpoint}\\nSettings: {CONFIG.to_dict()}")
if (RUN.report("search") / "search.csv").is_file():
    search_rows = list(csv.DictReader((RUN.report("search") / "search.csv").open()))
    for row in sorted(search_rows, key=lambda r: -float(r["best_map50"])):
        print(f"{row['trial']}: lr={row['lr']} weight_decay={row['weight_decay']} batch={row['batch']} "
              f"-> best AP50 {float(row['best_map50']):.4f} at epoch {row['best_epoch']}")
history = read_history(Path(checkpoint).parent / "history.csv")
plot_history(history, "Development")
"""))

# 8 Validation
add(md("""
## 8. Validation

Settings are chosen here and frozen for the test: the NMS IoU with the best AP50 and the operating confidence with
the best F1. Original and public validation images are reported separately to expose any domain gap (proposal
section 4.3).
"""))
add(code("""
validation = RUN.result("validation")
if validation is None:
    validation = validate(checkpoint, DATASET, RUN, DEVICE, batch=CONFIG.batch)
else:
    print("Loaded finished validation:\\n" + (RUN.report("validation") / "metrics.txt").read_text())
protocol = load_json(RUN.protocol_path)
settings = protocol["models"][MODEL_NAME]
print("NMS trials:", {row["nms_iou"]: round(row["map50"], 4) for row in protocol["nms_trials"][MODEL_NAME]})
print("Confidence trials (F1):", {row["confidence"]: round(row["micro_f1"], 4) for row in settings["confidence_trials"]})
print(f"Frozen settings: NMS IoU {settings['nms']['final_nms_iou']}, operating confidence {settings['operating_confidence']}")
plot_pr_curves(validation)
"""))
add(code("""
detector = YOLOv2Detector.from_checkpoint(checkpoint, DEVICE, protocol)
valid_truth = load_ground_truth(DATASET, "valid")
examples = pick_examples(validation)
shown = examples["success"][:2] + examples["missed_object"] + examples["background_or_low_overlap"]
if shown:
    show_predictions(detector, DATASET, "valid", valid_truth, shown)
"""))

# 9 Final test
add(md("""
## 9. Final test

Retrains from scratch on train + validation (450 images) for the development run's best epoch count, with every
setting frozen, then evaluates **once** on the 50 original test photos and measures inference speed (batch 1,
including preprocessing, decoding and NMS).
"""))
add(code("""
test = RUN.result("test")
if test is None and RUN_FINAL_TEST:
    test = final_test(RUN, DATASET, DEVICE)
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
    detector = YOLOv2Detector.from_checkpoint(RUN.final_checkpoint(), DEVICE, protocol)
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

*To be written from the results above:* how close each class came to its target and why; what the error analysis
shows (for example missed thin scissors versus background detections); whether original photos score below public
ones; and what would improve the detector next.
"""))

unused = {(module, name) for module, sections in SOURCES.items() for name in sections} - USED
if unused:
    raise SystemExit(f"Sections not placed in the notebook: {sorted(unused)}")
defined = {}
for module, name in USED:
    for node in ast.parse(SOURCES[module][name]).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in defined:
                raise SystemExit(f"{node.name} defined in both {defined[node.name]} and {module}")
            defined[node.name] = module

for number, cell in enumerate(CELLS):  # every code cell must be valid Python (shell lines aside)
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), f"notebook cell {number}", "exec")

notebook = {"cells": CELLS, "nbformat": 4, "nbformat_minor": 0,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT.name}: {len(CELLS)} cells, {len(USED)} source sections from {len(MODULES)} modules")
