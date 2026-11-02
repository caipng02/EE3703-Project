# Annotation and quality-control log

Prepared 2026-11-02 by Claude for team review. **Reviewer A has finished. An independent second review is still outstanding.**

## Originals (133 images, 297 boxes)

- **Method:** Two YOLOv8n detectors trained only on public images drafted boxes. Claude then set every box by eye on a gridded view. MobileSAM tightened a box only when its result stayed inside and agreed with the hand-set box. Of the 297 boxes, 263 use the tightened result and 34 keep the hand-set coordinates.
- **Check:** Every image was viewed with its final boxes. This led to 6 corrected boxes (originals 001, 030, 053, 086, 128, 132) and 1 added box (099).
- **Test images:** The 50 test images went through the same process. They were never used to train any model, including the drafting detectors. No accuracy has been measured on them.
- **Tool:** Boxes were not drawn in CVAT as the proposal describes. `original_data/annotations/originals_for_cvat.coco.json` and `original_data/working_images/` can be imported into CVAT for review.
- **To look at quickly:** `original_data/annotations/review_sheets/` shows all 133 images with their boxes.

### Labelling choices to confirm

- Drill boxes include the battery and the bit when one is fitted.
- A hand holding a tool is included only where it falls inside the tool's own extent.
- Partial tools are labelled when identifiable, including drill chucks at the frame edge in 094, 097, 099 and 122.
- Not labelled: hammers, pliers, saws, utility knives, scrapers, bit sets and other non-target tools.
- Not labelled because they are too ambiguous: blade tips on the wall rack in 092, 093, 095 and 096, and handles inside the storage tray in 097 and 099. The team may disagree on these.
- A tiny orange and red fragment at the edge of 098 is not labelled.

## Test split change

The test set was changed before annotation. Scene groups S3-01 (Worx drill) and S2-31, S2-32, S2-35 (steel bench) moved from test to development, so models see that drill and that bench in training. Blocks Q and F moved to test to keep 50 images. No model results influenced this. The earlier version is kept as `_archive/original_data_extras/*_v1*`.

## Public images (367 images)

- **Round 1:** Visual review of newly added images removed 32 images, including reciprocating saws, heat guns and glue guns labelled as drills, drawings, partial boxes and cooking shears.
- **Round 2:** Two-fold, out-of-fold detector disagreement flagged 121 images for visual review, and 8 were removed for label problems.
- **Open Images rotation:** Round 2 found that boxes are misaligned on photos with non-zero rotation metadata. All such candidates were excluded, 19 in total, 6 of which were in the build.
- **Replacements:** 6 more cluttered Working Tool frames were removed while reviewing replacements. All replacement images were viewed with their boxes.
- **Totals:** The exclusion list holds 65 entries (`_archive/scripts/data_selection/drop_keys.json`). No flagged image had its boxes edited; bad images were removed instead.

## Automated checks (`_archive/scripts/validate_final_dataset.py`)

All passed with 0 problems: 500 readable 640 x 640 images, matching images, COCO and YOLO files, valid classes and boxes, no image in two splits, no capture group crossing splits, test identical to the locked list, and test originals only.

The dHash check found no similar pair across splits (threshold 20 of 256 bits). Three Tool Monitoring images are similar to each other, all within one capture group.

## Augmentation check

`_archive/handoff_src/augment.py` implements the proposal's training-only list, plus moderate scale changes and partial occlusion for scissors. A rotation, shift or patch that would hide more than 70% of a box is discarded, so no box is lost; `_archive/scripts/test_data_pipeline.py` confirmed this over three passes of every training image. Boxes stayed aligned on both sample sheets. Mosaic is off by default (see `_archive/experiments/RESULTS.md`). If a team turns it on, a crop can still drop a box and leave a small unlabelled remnant.

## Baseline check (train and valid only, test not touched)

Pretrained YOLOv8n, 30 epochs, trained on `train/`. The best epoch was chosen on the same valid images, so these scores are optimistic. They show that the data can be learned, not what the from-scratch detectors will reach.

| Valid subset | Images | mAP@0.5 | Drill | Scissor | Screwdriver |
|---|---:|---:|---:|---:|---:|
| All | 92 | 0.896 | | | |
| Original | 17 | 0.944 | 0.995 | 0.953 | 0.884 |
| Public | 75 | 0.873 | 0.981 | 0.886 | 0.752 |

The original subset is small (42 boxes) and shows the same tool models as training. Screwdriver is the weakest class in both subsets.

## Team review

**Reviewer A (Jiali), 2026-11-02:** All 133 originals marked ok. Of the 76 public sample images, 75 marked ok and 1 marked fix.

**Reviewer B: not counted yet.** The B forms match the A forms exactly in every decision and note on all 209 rows, and the reviewer name is blank. They do not show an independent review, so a second team member still needs to complete them and enter their name.

| Image | Reviewer note | Outcome |
|---|---|---|
| `open_images_0218` | "only 1 scissor" | Confirmed: two near-identical boxes on one scissor. Duplicate removed. |
| `open_images_0097`, `open_images_0188` | "person" | Marked ok; a person holds the drill. No change. |
| `working_tool_0311` | "very similar to working_tool_0310" | Marked ok; both are in valid and the same capture group, so there is no leakage. Both kept. |

**Follow-up scan by Claude:** A search of all 500 label files for same-image box pairs with IoU of at least 0.7 found one more case, `open_images_0096`: two near-identical boxes on one scissor, confirmed by eye. Duplicate removed. This image was not in the review sample.

**Effect:** Two scissor boxes removed from train (180 to 178). Rebuilding changed no images, splits or other boxes. The checks passed with 0 problems. The proxy results in `_archive/experiments/` were not rerun, because 2 of 771 boxes are unlikely to change them. `_archive/public_data/statistics.json` still shows the earlier counts. The public sheets still draw both boxes on 0218, which look like one.

## Known limitations

- **Still required:** A second team member must independently review all 133 originals and at least 74 of the 367 public images (20%), using the B forms in `review/`. Record disagreements here.
- **Scene variety:** 95 public images come from 18 Working Tool videos and share tools and backgrounds. Originals show few distinct tool models.
- **Validation mix:** 54 of the 75 public valid images are Working Tool frames, because whole videos stay together. For model selection, also look at the 17 original valid images.
- **Licence:** PRACTICAS (11 images) is listed as CC BY 4.0 but its images look web-sourced. Replace them if the course requires certain provenance.
- **Image quality:** Hardware Tools images are 512 px sources, some upscaled. Set 1 originals are Telegram-compressed.
- **Screwdriver image count:** Box counts are balanced, but screwdrivers appear in fewer train images (110) than drills (169) or scissors (158), because many scenes hold several screwdrivers.
- **Proposal wording:** The proposal still states the older counts and source plan. Suggested wording is in `PROPOSAL_UPDATES.md`.
