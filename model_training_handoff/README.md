# Model training handoff

Everything needed to train and evaluate Faster R-CNN and YOLOv2 on the workbench tool dataset.

Use `EE3703_training.ipynb` on Colab with a GPU runtime. It holds all the code, and asks you to upload `dataset.zip`. Run all cells, then add the models in section 5.

- **Classes:** `0 drill`, `1 scissor`, `2 screwdriver`
- **Images:** 640 x 640, already letterboxed
- **Labels:** YOLO TXT in `labels/` and COCO JSON in `annotations.coco.json` (COCO `category_id` is the class plus 1)

| Split | Images | Originals | Public | Drill boxes | Scissor boxes | Screwdriver boxes |
|---|---:|---:|---:|---:|---:|---:|
| train | 358 | 66 | 292 | 174 | 178 | 181 |
| valid | 92 | 17 | 75 | 52 | 47 | 47 |
| test | 50 | 50 | 0 | 26 | 36 | 28 |

`dataset/manifest.csv` lists every image with its split and origin. `dataset/contract/` has 8 training images for checking that your loader reads boxes correctly.

## What the notebook handles

- **Loading:** `train_loader` and `valid_loader` give normalized images `[B, 3, 640, 640]` with pixel boxes `[x1, y1, x2, y2]`. YOLOv2 can convert them with `boxes_to_yolo`.
- **Normalization, augmentation and sampling:** these are already applied, so don't add your own.
- **Box check:** section 4 draws boxes so you can confirm they sit on the right tools.
- **Scoring:** not included, because this belongs to the evaluation lead. The scorer must report original and public validation images separately, using `origin` in `dataset/manifest.csv`, and use the same NMS rule for both detectors.

## Rules

1. **Use validation for every decision:** epochs, thresholds, anchors, hyperparameters and model choice. Compute YOLOv2 anchors from `train` only.
2. **Leave the test set alone until the end.** Fix every setting first, then evaluate once. Section 5 of the notebook shows how to load the test split.
3. **Retrain before the final test.** Train on train and valid together with the settings fixed on validation (code in section 5). No validation data is left at that point, so fix the epoch count beforehand.

## Starting settings to recheck

`MOSAIC_P = 0.0` and `ORIGINAL_WEIGHT = 1.0` are in the notebook's data loading cell. They come from a pretrained stand-in detector, where alternatives showed no measurable gain, so recheck them on your detector using validation.

If AP on the original valid images is substantially below AP on the public valid images, raise `ORIGINAL_WEIGHT` (try 2.0) and reevaluate on validation.

