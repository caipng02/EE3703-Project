# EE3703 Person 3: YOLOv2 / ResNet-18

## Current checkpoint

`best.pt` is the completed epoch-60 checkpoint from **eight training images**.
It starts from random initialization, with positive-objectness weight 5 and
background weight 0.5. Training took 38.25 seconds on the recorded Colab run.

| Training-subset metric | Value |
|---|---:|
| mAP@0.50 | 95.13% |
| Drill AP50 | 95.00% |
| Scissor AP50 | 100.00% |
| Screwdriver AP50 | 90.38% |

These are small-training-subset diagnostics using custom all-points AP, not
validation or final-test results. Full-data training has not been completed.
Keep the 50-image test split untouched until the final protocol is fixed.

## Files

- `YOLO_Project_round2_2_obj5.ipynb`: self-contained model, loader, decoder and training code, with recorded run outputs and images. Training switches are disabled for browsing.
- `best.pt`: inference weights; no separate model Python file is required.
- `model_metadata.json`: exact input/output conventions and checkpoint SHA256.
- `results/`: original run summary, history, metrics and two prediction images.

## Load the checkpoint

Run the notebook definition cells before loading the weights. All optional
training gates default to False. Adjust PROJECT_DIR to your own mounted Drive
location if using the notebook's data preparation. Loading the checkpoint alone
does not require training or running the final-test cell.

```python
checkpoint_path = Path(
    "/content/drive/MyDrive/EE3703GroupProject_person3_yolov2/"
    "integration_round2_2_20261104/best.pt"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
assert tuple(checkpoint["classes"]) == CLASSES
model = YOLOv2ResNet18(checkpoint["anchors"])
model.load_state_dict(checkpoint["state_dict"], strict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print("Loaded checkpoint epoch:", checkpoint["epoch"])
```

## Input and output contract for P4

- Classes remain **0=drill, 1=scissor, 2=screwdriver**. Do not add one before
  passing canonical predictions to P4; its COCO adapter performs the conversion.
- Input is RGB, EXIF-transposed, letterboxed to640x640 with fill114, divided by255,
  then normalized with mean[0.485,0.456,0.406] and std[0.229,0.224,0.225].
  Existing dataset images are already640x640; avoid inconsistent preprocessing.
- `postprocess()` returns normalized **xyxy** boxes. Convert to640-canvas pixel
  coordinates before passing to P4. Original camera-resolution coordinates need
  the letterbox transformation reversed instead.
- Export at score floor0.001 with class-aware NMS IoU0.5 and max300 detections.
  Use filename-only image IDs, include empty detections and follow P4's wrapper
  and metadata schema. Use P4's COCO evaluator for shared model comparisons.
- Example screenshots use display threshold0.05 and therefore show many weak
  boxes. A display threshold0.25 is clearer on these eight training images, but
  the formal operating threshold must be selected using validation data.

## Data

Use the team's existing `dataset.zip` and fixed splits: train358, validation92,
test50. The raw dataset is not duplicated in this package. Refer to Person1's
shared data handoff. This notebook uses its own mild flip/brightness/contrast
augmentation and class sampler; align that protocol with the team for comparisons.
