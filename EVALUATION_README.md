# Detector evaluation handoff

Use Python 3.10+ and install the AP scorer with `python -m pip install pycocotools`. Point `--dataset-root` at Person 1's unpacked `dataset` directory from the [data branch](https://github.com/caipng02/EE3703-Project/tree/data).

Export decoded detections after class-aware NMS. Use pixel `[x1, y1, x2, y2]` on the saved 640×640 canvas and class IDs `0=drill`, `1=scissor`, `2=screwdriver`. Include exactly one record per split image, with empty arrays for zero detections. Use the actual filename, including its extension.

Example prediction JSON excerpt:

```json
{
  "data_kind": "real",
  "model_name": "YOLOv2",
  "metadata": {
    "model_family": "yolov2",
    "checkpoint": "validation_checkpoint.pt",
    "nms_stage": "final",
    "nms_iou": 0.50,
    "nms_rule": "class_aware",
    "export_score_floor": 0.001
  },
  "images": [
    {"image_id": "example.jpg", "boxes": [[20, 30, 100, 150]], "labels": [0], "scores": [0.9]}
  ]
}
```

Run validation with one export per model if NMS is already selected:

```text
python person4_evaluation.py --dataset-root path/to/dataset --dataset-split valid --project-protocol --predictions frcnn_valid.json yolo_valid.json --output validation_results
```

For the NMS sweep, add `--select-nms` and pass one export per candidate. Use the same model name, checkpoint, and export floor within each model's sweep. Set `model_family` to `faster_rcnn` (IoUs 0.50 and 0.70) or `yolov2` (0.45, 0.50 and 0.60). If Faster R-CNN sweeps proposal NMS, use `nms_stage: "rpn"` and record the fixed `final_nms_iou` separately. Use an actual export floor of 0.001 or lower with the default AP floor.

Save `validation_results/evaluation_protocol.json`. The final test reuses the selected settings. Keep the model names and NMS settings consistent in the test exports.

```text
python person4_evaluation.py --dataset-root path/to/dataset --dataset-split test --allow-test --project-protocol --frozen-protocol validation_results/evaluation_protocol.json --predictions frcnn_test.json yolo_test.json --output final_results
```

Open `index.html` for metrics, error records, and overlays. `comparison.csv` has model-level scores and `subsets.csv` has original/public breakdowns.
