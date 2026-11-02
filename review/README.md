# Annotation review kit

The proposal (section 2.3) requires two team members to independently review every original annotation and at least 20% of the public annotations. Claude's annotations do not count as this review.

## How to review

1. Reviewers A and B each take their own forms. Do not look at each other's answers until both are finished.
2. Open each image with its boxes, either in the sheets or in CVAT.
   - Originals: `original_data/annotations/review_sheets/`
   - Public sample: `public_sheets/`
3. Colours: green is drill, yellow is scissor, cyan is screwdriver.
4. For each row, enter a reviewer name and a decision (`ok`, `fix` or `remove`), and describe any problem.
5. Compare the two forms, resolve disagreements together, and record the outcome in `review/QC_LOG.md`.
6. Give the agreed fixes to whoever maintains the dataset, then rebuild and rerun the checks (see `_archive/REBUILD.md`).

## What to check

- Every visible drill, scissor and screwdriver has a box, including partly hidden ones.
- Each box has the right class and fits the tool closely.
- Drill boxes include the battery and any fitted bit.
- Non-target tools such as hammers, pliers, saws and knives have no box.
- Originals only: decide the open cases listed in `review/QC_LOG.md`, such as the wall-rack blades.

## Importing into CVAT

Create a task with labels `drill`, `scissor` and `screwdriver`, upload the images, then use Upload annotations with format COCO 1.0.

| Set | Images | Annotations |
|---|---|---|
| Originals | `original_data/working_images/` | `original_data/annotations/originals_for_cvat.coco.json` |
| Public sample | `review/public_sample/images/` | `review/public_sample/public_sample_for_cvat.coco.json` |

Rename the annotation file to `instances_default.json` and put it in an `annotations` folder inside a zip if CVAT asks for its usual archive layout.

## Keeping the test set clean

Fifty of the originals are locked test images (marked `test` in the forms). Fix their labels only for real annotation errors, and never because of how a model performed on them.
