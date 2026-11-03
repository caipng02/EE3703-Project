# EE3703 Person 3: YOLOv2 / ResNet-18 - baseline in progress

## Current result (3 November 2026)

A real-data baseline ran on Colab Tesla T4 with 358 training and 92 validation
images for 10 epochs. The separate 50-image original test split was not evaluated.

- Training loss: 61.6297 in epoch 1, 12.3470 in epoch 10.
- Validation mAP@0.50: **0.0 in every epoch**.
- Best checkpoint epoch: **1**. The current code updates the checkpoint only
  when validation mAP strictly improves; it does not save epoch 10 as `best.pt`
  when all validation scores tie at zero.
- Training time reported by the notebook: 237.35 seconds.
- Settings: learning rate 1e-4, weight decay 1e-4, batch size 2.
- Earlier smoke run: 32 training images, 2 epochs; no validation evaluation.

The training pipeline executes, but useful detection performance has not been
established. This is a progress/debugging handoff for integration, not a final
trained-model delivery. Hyperparameter search and final retraining were skipped.

## Files

- `YOLO_Project.ipynb`: recorded Colab notebook. All saved execution outputs and
  three image outputs are preserved: training annotation preview, smoke loss
  curve, and baseline training/validation curves. These are not detector
  prediction images. Outdated narrative text was updated; executable code was
  not changed and no training was rerun while packaging.
- `yolov2_person3.py`: reusable model/data/training definitions, consistent with
  the notebook's core implementation. It does not contain the added notebook
  cells that mount Drive, convert the team manifest or launch the baseline.
- `manifest.template.json`: empty schema example, not the populated dataset
  manifest and not the dataset itself.

**No checkpoint is included.** The recorded run saved `best.pt`, `history.json`,
`summary.json` and `training_curves.png` to the Drive `baseline_10epochs` directory
printed in the notebook. Retrieve `best.pt` separately if required for debugging,
and identify it as the epoch-1 baseline checkpoint with zero validation mAP.

## Dataset and execution

Class order is `0=drill`, `1=scissor`, `2=screwdriver`.
Use the team's data archive from the `data` branch:
https://github.com/caipng02/EE3703-Project/tree/data/model_training_handoff

The notebook converts the team's `manifest.csv` into its own `manifest.json`
schema and reads existing 640 x 640 images / YOLO TXT labels through its own
loader. It does not use the separate team notebook's online augmentation and
sampling pipeline; verify those differences before comparative experiments.

Before rerunning, set `PROJECT_DIR` to the receiving user's own Drive folder
containing `dataset.zip`. A GPU runtime is required for training. Running the
preparation cell creates a new temporary extraction and a new Drive run directory.

The appended smoke and baseline cells **run training when executed**, regardless
of the three RUN_SEARCH / RUN_TRAINING / RUN_GOLD_TEST flags. Do not select Run All
just to inspect outputs. The original search / final-retraining / final-test
sections remain disabled in the recorded execution.

The synthetic self-test's `real_training` output is a legacy fixed helper string,
not a live project-status field. The baseline's saved logs are the source for
real training progress and metrics.

## Next checks with the team

Investigate confidence filtering, predictions and target coordinate/class handling,
evaluation behavior and learning convergence on train/validation data. Inspect
actual validation predictions, agree on the shared scoring protocol, and report
original/public validation performance separately. Keep the final test split
untouched until settings and final retraining are fixed.

AP uses all-points interpolation, not the official COCO 101-point evaluator.
