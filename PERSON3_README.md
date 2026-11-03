# Person 3: YOLOv2 with ResNet-18 (work in progress)

Initial implementation for the workbench-tool detection project. This is a
preparation-stage handoff, not a trained detector or a final course submission.

## Files

- `YOLO_Project.ipynb`: self-contained Colab notebook with model definitions,
  synthetic checks, and gated training and evaluation sections.
- `yolov2_person3.py`: Python source for the same model and preparation workflow.
- `manifest.template.json`: empty template for this implementation's JSON-based
  dataset interface; it does not contain the team's annotations.

The code uses the class order `0=drill`, `1=scissor`, `2=screwdriver` and
randomly initialized weights. Training, hyperparameter search, and final-test
execution are disabled by default in the notebook. Synthetic checks passed on
31 October 2026 in the original local preparation run; they do not measure real
detector accuracy. The notebook outputs in this handoff copy were cleared to
remove outdated execution and dataset-status information. The original local
notebook and its saved outputs are unchanged.

## Team data integration remains to be done

The team's [data handoff](https://github.com/caipng02/EE3703-Project/tree/data/model_training_handoff)
describes a labeled dataset and existing data loaders. This code was written
against its own `manifest.json` and dataset workflow; it is **not yet verified
as compatible** with the team's `manifest.csv` or loaders. Before real training:

1. Check the current team notebook, data splits, label mapping, and provenance.
2. Implement or agree on the interface, including image normalization, box
   coordinates, augmentation, and sampling. Avoid applying preprocessing twice.
3. Visually check boxes on the team's contract images and run a small batch.
4. Select settings using training/validation data only. Preserve the team's
   held-out test set for the final fixed-configuration evaluation.

No real-data training, detector metrics, trained checkpoints, or inference
results are included in this handoff. Coordinate the shared evaluation protocol
and final integration with the team.
