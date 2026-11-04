# Development record

What was changed and run for the YOLOv2 and Faster R-CNN detectors, with the results and known limitations.
YOLOv2 comes first; Faster R-CNN has its own section after the YOLOv2 limitations. Scores are
**validation** unless marked otherwise (92 images: 17 original, 75 public). Metric: COCO AP50 averaged over
drill, scissor and screwdriver, from pycocotools.

## YOLOv2 results summary

| Trial | Date | Backbone | Recipe | Best val AP50 | Notes |
|---|---|---|---|---|---|
| 0 | 3 Nov | random | Person 3's notebook: light augmentation, batch 2, lr 1e-4, 10 epochs | 0.000 | Person 3's own pipeline |
| 1 | 4 Nov | random | Full augmentation, batch 2, lr 1e-4, 20 epochs | 0.041 (epoch 16) | First integrated run |
| 2A | 4 Nov | random | Batch 16, lr 1e-3, objectness weight 5 | 0.028 (epoch 10) | Abandoned after 15 epochs: no better than trial 1 |
| 3 | 4 Nov | ImageNet | Round-3 recipe (below), 60 epochs max, patience 10 | 0.738 | Run `yolov2_20261104_0126`; selected until trial 5b |
| 4 | 4 Nov | ImageNet | Trial 3 + 9 anchors + objectness target 1 | 0.722 (epoch 50) | Run `screwdriver_20261104_0246`; screwdriver experiment |
| 5 | 4 Nov | ImageNet | Trial 3 + 40 × 40 grid (stride 16) + 9 anchors | 0.039 (epoch 7) | Run `finegrid_20261104_0537`; early-stopped at epoch 17, never started learning |
| **5b** | **4 Nov** | **ImageNet** | **Trial 5 with peak lr 5e-4 and 5 warmup epochs** | **0.803 (epoch 45)** | **Run `finegrid2_20261104_0654`; selected; test AP50 0.933** |
| 5c | 4 Nov | ImageNet | Trial 5b + objectness target 1 | 0.788 (epoch 42) | Run `finegrid3_20261104_0813`; screwdriver up, overall down |
| 5d | 4 Nov | ImageNet | Trial 5b + two anchors per object | 0.801 (epoch 29) | Run `finegrid4_20261104_0934`; tied overall, screwdriver up, scissor and originals down |

Trial 3 recipe: ImageNet-initialized ResNet-18 backbone, full augmentation + mosaic 0.5, class-aware sampling,
batch 8, AdamW lr 1e-3 / weight decay 1e-4, 3-epoch warmup + cosine decay, weight EMA, objectness prior 1%,
positive-objectness weight 5, 5 anchors.

## Trial 3 in detail (run `yolov2_20261104_0126`, Colab T4)

**Training.** Validation AP50 (EMA weights) rose from 0.04 (epoch 1) to 0.70 by epoch 37 and peaked at
**0.735 at epoch 45**; early stopping after epoch 55. Validation loss fell steadily from 22.4 to 11.8, with no
sign of overfitting. Training loss stays above validation loss because training images are augmented.

**Validation with the selected settings** (NMS IoU 0.45, operating confidence 0.4):

| | AP50 | Proposal target | AP50:95 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Overall** | **0.738** | 0.70 ✅ | 0.332 | 0.728 | 0.753 | 0.741 |
| Drill | 0.943 | 0.75 ✅ | 0.452 | 0.907 | 0.942 | 0.925 |
| Scissor | 0.835 | 0.65 ✅ | 0.379 | 0.830 | 0.830 | 0.830 |
| Screwdriver | **0.434** | 0.70 ❌ | 0.166 | 0.440 | 0.468 | 0.454 |

- Original photos AP50 **0.781** (17 images, 42 objects); public images 0.711 (75 images, 104 objects).
- Mean IoU of matched detections 0.741; 110 true positives, 41 false positives, 36 missed objects.
- NMS trials: 0.45 → 0.7376, 0.50 → 0.7367, 0.60 → 0.7350 (almost no effect).
- Confidence trials (F1): 0.2 → 0.604, 0.3 → 0.705, 0.4 → 0.741.
- Errors at confidence 0.4: missed objects 36, poor localization 21, background detections 16, wrong class 3,
  duplicates 1.
- Model: 37.2 M parameters (backbone 11.2 M, YOLOv2 layers 26.0 M), checkpoint 149 MB.

**What made the difference.** From trial 1 to trial 3, validation AP50 went from 0.041 to 0.738. Trial 3
changed the backbone initialization and the training recipe together, so the share of each is not measured.
The from-scratch runs plateaued early (training loss flat from epoch 3), which points to the pretrained backbone
as the main factor.

## Screwdriver analysis

Box statistics from the dataset labels:

| | Drill | Scissor | Screwdriver |
|---|---|---|---|
| Median box, train (px) | 303 × 297 | 216 × 217 | **108 × 94** |
| Median box, validation (px) | 265 × 256 | 155 × 157 | **73 × 86** |
| Boxes with a side under 32 px (train) | 0% | 1% | **22%** |
| Long/short side above 3 (train) | 3% | 7% | **49%** |
| Median IoU with the best anchor (train) | 0.71 | 0.64 | **0.47** |
| Objects sharing a grid cell (train) | 0% | 0% | 0% |

Likely causes: screwdrivers are about a third the size of drills and often smaller than one 32-px grid cell on
the single 20 × 20 output grid; none of the 5 anchors matches a small, thin shape; and the IoU objectness target
pushes small objects (which reach lower IoUs) towards low confidence, so many fall below the 0.4 threshold
(recall 0.47). Classes are almost never confused (3 wrong-class errors).

### Trial 4: screwdriver experiment (run `screwdriver_20261104_0246`)

Trial 3's recipe with two changes aimed at small, thin objects: `anchors: 9` (k-means with 9 shapes, adding thin
ones such as 95 × 30 px and 45 × 95 px) and `objectness_target: one` (objectness target 1 instead of the IoU of the
predicted box). The final test of trial 3 was stopped during its retrain, before any test image was evaluated, so
the choice between the two models is made on validation only. Best epoch 50 of 60; selected NMS IoU 0.50,
confidence 0.4.

| Validation | Trial 3 | Trial 4 | Change |
|---|---|---|---|
| Overall AP50 | **0.738** | 0.722 | −0.016 |
| AP50:95 | **0.332** | 0.287 | −0.045 |
| Precision / recall | **0.728** / 0.753 | 0.583 / **0.795** | |
| F1 (confidence 0.4) | **0.741** | 0.672 | −0.069 |
| False positives / missed objects | **41** / 36 | 83 / **30** | |
| Drill AP50 | **0.943** | 0.897 | −0.046 |
| Scissor AP50 | **0.835** | 0.785 | −0.050 |
| Screwdriver AP50 | 0.434 | **0.484** | +0.050 |
| Screwdriver precision / recall | **0.440** / 0.468 | 0.364 / **0.681** | |
| Original / public AP50 | **0.781** / **0.711** | 0.776 / 0.696 | |

**Finding.** The changes help the weak class: screwdriver AP50 +0.05 and recall from 47% to 68%. But they make
the detector confident about poor boxes too: false positives double, precision and box tightness (AP50:95) fall,
and drills and scissors lose about 0.05 AP50 each. F1 was still rising at the top of the confidence grid
(0.519, 0.616, 0.672), so higher-confidence boxes would trade some of that recall back. Both changes were made
together, so their separate effects are not measured; the thin anchors are the likely source of the screwdriver
gain and the objectness target of the extra false positives. A 9-anchors-only run would separate them.

**Decision (after trial 4).** Trial 3 is kept for the final test: it has the higher overall validation AP50, the proposal's
selection criterion, and is better on precision, F1, AP50:95 and two of three classes.

### Trial 5: finer output grid (run `finegrid_20261104_0537`)

Screwdriver AP50 must reach 0.70. The analysis points to resolution: on the 20 × 20 grid each cell is 32 px, while
validation screwdrivers have a median box of 73 × 86 px and many are thin. Trial 5 predicts on a 40 × 40 grid
(stride 16): the deep stride-32 features are upsampled and joined with the stride-16 features before the same
head, so the backbone, ImageNet weights, anchors, loss and NMS are unchanged and the detector keeps one output
scale. It keeps trial 4's 9 anchors (the likely source of its screwdriver gain) and returns to the IoU
objectness target (the likely source of its false positives). Cost: about 1.3× the compute per image. With
115,200 anchor slots per batch of 8, background makes up 98% of the initial loss without the objectness prior.
Whichever of trials 3 and 5 has the higher overall validation AP50 goes to the final test.

**Result.** Training never got going. Validation AP50 (EMA weights) peaked at 0.038 at epoch 7, and early
stopping (patience 10) ended the run after epoch 17 at 0.037. Loss was unstable around the end of warmup, when
the learning rate peaks: validation loss jumped to 46,762 at epoch 3 and 511 at epoch 8 (otherwise about 17),
and training loss swung between 37 and 179.

| Validation | Trial 3 | Trial 5 |
|---|---|---|
| Overall AP50 | **0.738** | 0.039 |
| AP50:95 | **0.332** | 0.008 |
| Drill / scissor / screwdriver AP50 | **0.943** / **0.835** / **0.434** | 0.088 / 0.028 / 0.001 |
| Original / public AP50 | **0.781** / **0.711** | 0.035 / 0.054 |
| Detections at confidence 0.2–0.4 | | none (F1 0, all 146 objects missed) |

**Reading.** This does not show that the finer grid is worse. It shows the run stopped before it learned
anything. The 9-anchor runs start slowly: trial 4 was also below 0.05 AP50 for its first 9 epochs, reached only
0.09 by epoch 17, and passed 0.15 after epoch 20, while trial 3 had 0.46 by epoch 17. Trial 5 adds 4× the
anchor slots and a bigger spike at peak learning rate (validation loss 46,762, against 47 in trial 4), so its
slow start lasted longer than the 10-epoch patience. A fair retry would need a gentler start, for example a
lower peak learning rate or a longer warmup, or early stopping that only counts epochs after warmup. Both are
changes to the training recipe, not to the grid.

**Decision (after trial 5).** Trial 3 stays the model for the final test (validation AP50 0.738 against
0.039).

### Trial 5b: finer grid with a gentler start (run `finegrid2_20261104_0654`)

Trial 5's settings with two changes to the start of training only: peak learning rate 5e-4 instead of 1e-3, and a
5-epoch warmup instead of 3. Notebook `experiments/YOLO_finegrid_experiment.ipynb` (trial 5's outputs are kept in
`experiments/YOLO_finegrid_experiment_run1.ipynb`).

**Training.** No loss spike this time: validation loss fell smoothly from 19.6 to 12.8. Validation AP50 (EMA
weights) passed 0.1 at epoch 9, 0.5 at epoch 18 and 0.7 at epoch 25, and peaked at **0.803 at epoch 45**; early
stopping after epoch 55. Selected NMS IoU 0.50 (trials: 0.45 → 0.7996, 0.50 → 0.8028, 0.60 → 0.7967) and
operating confidence 0.3 (F1: 0.2 → 0.724, 0.3 → 0.814, 0.4 → 0.809).

| Validation | Trial 3 | Trial 5b | Change |
|---|---|---|---|
| Overall AP50 | 0.738 | **0.803** | +0.065 |
| AP50:95 | 0.332 | **0.367** | +0.035 |
| Operating confidence | 0.4 | 0.3 | |
| Precision / recall | 0.728 / 0.753 | **0.835** / **0.795** | |
| F1 | 0.741 | **0.814** | +0.073 |
| True positives / false positives / missed | 110 / 41 / 36 | **116** / **23** / **30** | |
| Mean IoU of matched detections | **0.741** | 0.733 | |
| Drill AP50 | **0.943** | 0.923 | −0.020 |
| Scissor AP50 | 0.835 | **0.880** | +0.045 |
| Screwdriver AP50 | 0.434 | **0.605** | +0.171 |
| Screwdriver precision / recall | 0.440 / 0.468 | **0.718** / **0.596** | |
| Original / public AP50 | 0.781 / 0.711 | **0.820** / **0.799** | |

Against the proposal's targets: overall 0.803 (0.70 ✅), drill 0.923 (0.75 ✅), scissor 0.880 (0.65 ✅),
screwdriver 0.605 (0.70 ❌, gap down from 0.27 to 0.10). Same model size (37.3 M parameters, 149 MB checkpoint).

**Finding.** The finer grid, with thin anchors, does what the screwdriver analysis predicted: screwdriver AP50
+0.17, with both precision and recall up, and unlike trial 4 it does not cost the other classes much (drill −0.02,
scissor +0.05) or add false positives (23, down from 41). Trial 5b differs from trial 3 in four settings (grid,
anchors, learning rate, warmup), so the share of each is not measured; the lower learning rate alone was not tried
on the 20 × 20 grid.

**Decision (after trial 5b).** Trial 5b replaces trial 3 for the final test: it has the higher overall validation
AP50 (0.803 against 0.738), the proposal's selection criterion, and is better on every overall measure.

### Trial 5c: trial 5b + objectness target 1 (run `finegrid3_20261104_0813`)

Trial 5b with one change, `objectness_target: one`, aimed at screwdriver recall: trial 5b missed screwdrivers more
than it mislabelled them (precision 0.718, recall 0.596), and the IoU target pushes small, thin boxes towards low
confidence. Notebook outputs kept in `experiments/YOLO_finegrid_experiment_run3.ipynb`.

**Training.** Smooth, no spikes; validation AP50 peaked at 0.788 at epoch 42, early stopping after epoch 52.
Validation loss bottomed out at epoch 36 (14.51) and rose slightly afterwards. Selected NMS IoU 0.50 and
operating confidence 0.4 (F1: 0.2 → 0.685, 0.3 → 0.743, 0.4 → 0.760).

| Validation | Trial 5b | Trial 5c | Change |
|---|---|---|---|
| Overall AP50 | **0.803** | 0.788 | −0.015 |
| AP50:95 | **0.367** | 0.347 | −0.020 |
| Precision / recall | **0.835** / **0.795** | 0.740 / 0.781 | |
| F1 | **0.814** | 0.760 | −0.054 |
| True positives / false positives / missed | **116** / **23** / **30** | 114 / 40 / 32 | |
| Drill AP50 | **0.923** | 0.879 | −0.044 |
| Scissor AP50 | **0.880** | 0.822 | −0.058 |
| Screwdriver AP50 | 0.605 | **0.664** | +0.059 |
| Screwdriver precision / recall | **0.718** / 0.596 | 0.593 / **0.681** | |
| Original / public AP50 | 0.820 / **0.799** | **0.830** / 0.762 | |

**Finding.** The objectness target has the same effect as in trial 4, now measured on its own: screwdriver recall
rises (0.60 → 0.68) and screwdriver AP50 gains 0.06, but the detector also becomes confident about poor boxes,
so false positives almost double (23 → 40) and drills and scissors each lose about 0.05 AP50. The screwdriver
target (0.70) is still not met, and the overall score falls.

**Decision (after trial 5c).** Trial 5b stays the model for the final test (overall validation AP50 0.803
against 0.788).

### Trial 5d: trial 5b + two anchors per object (run `finegrid4_20261104_0934`)

Trial 5b with one change, `anchors_per_object: 2`: each object trains the two best-matching free anchors in its
cell instead of one, so thin screwdrivers, which fit no single anchor well, get twice the positive examples while
the IoU objectness target is kept (trial 5c's target of 1 had doubled false positives). The loss-demonstration
cell confirmed the change: 66 assigned anchors for the 8 sample images, against 33 in trial 5b. Notebook outputs
kept in `experiments/YOLO_finegrid_experiment_run4.ipynb`.

**Training.** Faster and smooth: validation AP50 passed 0.5 at epoch 11 (trial 5b: 18) and 0.7 at epoch 16
(trial 5b: 25), peaked at 0.800 at epoch 29, and early stopping ended the run after epoch 39. Losses are about twice
trial 5b's because every object has two positive anchors, so they are not comparable across the two trials.
Selected NMS IoU 0.45 (0.45 → 0.8008, 0.50 → 0.8002, 0.60 → 0.7969) and operating confidence 0.4 (F1: 0.2 → 0.735,
0.3 → 0.772, 0.4 → 0.808).

| Validation | Trial 5b | Trial 5d | Change |
|---|---|---|---|
| Overall AP50 | **0.803** | 0.801 | −0.002 |
| AP50:95 | 0.367 | **0.377** | +0.010 |
| Precision / recall | **0.835** / 0.795 | 0.823 / 0.795 | |
| F1 | **0.814** | 0.808 | −0.006 |
| True positives / false positives / missed | 116 / **23** / 30 | 116 / 25 / 30 | |
| Mean IoU of matched detections | 0.733 | **0.752** | |
| Drill AP50 | 0.923 | **0.960** | +0.037 |
| Scissor AP50 | **0.880** | 0.801 | −0.079 |
| Screwdriver AP50 | 0.605 | **0.642** | +0.037 |
| Screwdriver precision / recall | **0.718** / 0.596 | 0.689 / **0.660** | |
| Original / public AP50 | **0.820** / 0.799 | 0.773 / **0.827** | |

**Finding.** Two anchors per object raise screwdriver recall (0.60 → 0.66) without trial 5c's false positives
(25 against 23), and give tighter boxes (AP50:95 +0.01, mean matched IoU 0.73 → 0.75). But scissors lose 0.08
AP50 and the overall score is unchanged: the difference of 0.002 is far below what 92 images can resolve. The
screwdriver target (0.70) is still not met. On the 17 original photos, the kind of image that makes up the test
set, trial 5d scores lower (0.773 against 0.820), though 17 images are too few to rely on.

**Decision (after trial 5d).** Trial 5b stays the model for the final test, by the selection rule fixed before
these trials (higher overall validation AP50, 0.803 against 0.801), and it also scores higher on original photos.
Development stops here: further trials chosen on the same 92 validation images would make the validation score
increasingly optimistic.

## Final test (run `finegrid2_20261104_0654`, 4 Nov, Colab T4)

Trial 5b's settings, retrained from scratch on train + validation (450 images, 57 updates per epoch) for 45 epochs
(trial 5b's best epoch), then evaluated once on the 50 original test photos. NMS IoU 0.50 and operating confidence
0.30 were frozen from validation. Training loss fell from 43.0 to about 25 (26.3 at epoch 45).

The run folder held an earlier retrain that had been stopped before finishing. Only `work/final_train` was cleared;
no test folder existed, so no test image had been evaluated before this single test.

| Test (50 originals, 90 objects) | AP50 | Target | AP50:95 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Overall** | **0.933** | 0.70 | 0.447 | 0.889 | 0.889 | 0.889 |
| Drill (26) | 0.906 | 0.75 | 0.448 | 0.955 | 0.808 | 0.875 |
| Scissor (36) | 0.943 | 0.65 | 0.462 | 0.943 | 0.917 | 0.930 |
| Screwdriver (28) | 0.950 | 0.70 | 0.431 | 0.788 | 0.929 | 0.852 |

TP 80, FP 10, FN 10. Speed: 41.9 ms per image (23.9 FPS) at batch 1 on the T4, including preprocessing, decoding
and NMS. Model: 37,274,760 parameters, 149.2 MB checkpoint. All 50 annotated test photos are saved in the run's
`test_predictions/` folder for the demo video.

**Reading the result.** Every class beats its target. The test score is well above validation (0.933 against
0.803), most visibly for screwdrivers (0.950 against 0.605). Three things contribute, and they cannot be separated
with this data:

- **Domain:** the test holds only original photos. On validation, originals already scored higher than public
  images (0.820 against 0.799), and the 17 validation originals are the closest match to the test.
- **More training data:** the final model trained on 450 images instead of 358, including the validation originals.
- **Small test set:** 90 objects (28 screwdrivers), so one or two detections move a class's AP by several points.
  The test score is unbiased but uncertain.

Localization stays the weak point: AP50:95 is 0.447 and the mean matched IoU 0.74, so boxes are found but often
loose. Screwdrivers have the lowest precision (0.788): 7 of the 10 false positives are screwdriver detections (drill 1,
scissor 2, from each class's precision and recall).
The error figure in the notebook breaks these down.

## YOLOv2 limitations

- **Screwdrivers on validation** stayed below target (AP50 0.605 vs 0.70 with trial 5b, up from 0.434 with trial
  3), though they reach 0.950 on the test originals; see the analysis above and the final test. Objectness target 1
  (trial 5c) raised the validation score to 0.664 and two anchors per object (trial 5d) to 0.642, but neither
  improved the overall score.
- **Loose boxes:** AP50:95 is 0.37 on validation and 0.45 on test (trial 5b), so detections find the tools but their boxes are often too
  large or offset.
- **Small validation set:** 92 images, only 17 originals, so per-class and per-origin numbers are noisy; the best
  epoch was also chosen on validation, so validation AP is slightly optimistic. Choosing the best of several
  trials on the same 92 images adds to that.
- **Confidence grid ceiling** (trial 3): the proposal's grid stops at 0.4 and trial 3's F1 was still rising
  there. Trial 5b peaks inside the grid (0.3), so this no longer limits the selected model.
- **No hyperparameter search yet:** the proposal's random search over learning rate, weight decay and batch
  size (section 4.2) is implemented (`SEARCH_CONFIGS`) but each trial used one configuration.
- **Several trials chosen on one validation set:** trials 3, 4, 5, 5b, 5c and 5d were all compared on the same
  92 images, so the selected model's validation score is optimistic; the test score is the unbiased one.
- **Bundled changes:** trial 3 changed the backbone initialization and several training settings at once; there
  is no pretrained-vs-random ablation with the same recipe. Trial 5b likewise changed the grid, anchors, learning
  rate and warmup together.
- **Final retrain is unverified:** the model trained on train + validation cannot be scored on validation, and
  its cosine schedule spans 45 epochs rather than part of 60.
- **First learning-rate step skipped on GPU:** the mixed-precision scaler skips its first update, which PyTorch
  reports as `lr_scheduler.step()` before `optimizer.step()`; 1 step out of about 2,500, negligible.
- **Faster R-CNN** is compared in the next section; its final test is still to be run.

## Faster R-CNN

### Starting point: Person 2's branch (`RCNN_debug`)

Person 2's Faster R-CNN: a ResNet-18 backbone (ImageNet weights) whose `dense` profile stops at layer 3 (stride 16,
a 40 × 40 feature map), anchors of 5 sizes (32–512 px) × 3 ratios (0.5, 1, 2), an RPN, and an RoI head
(RoIAlign 5 × 5, one 1024-unit layer, 4-way classifier and class-specific box offsets). Training: SGD lr 5e-4
(heads 4×), batch 2, 10 epochs, constant learning rate; per image 300 proposals, 32 sampled RoIs (at most 8
foreground). Person 2's measured development run, from their branch's README:

| Validation (Person 2's run) | Value |
|---|---|
| Overall AP50 | 0.266 (0.248 for their earlier three-size baseline) |
| Screwdriver AP50 | 0.007 |
| Screwdrivers covered by a proposal at IoU ≥ 0.5: top 100 / top 600 | 2 / 13 of 47 |
| Screwdrivers the RoI head names correctly when given the exact box | 40 of 47 |

So the classifier mostly worked; objects were lost in the RPN, which rarely proposed a box around a screwdriver.

### Integration (4 Nov)

The model, losses and decoding were moved unchanged into the package (`models/faster_rcnn.py`); the RPN NMS IoU
became a parameter so validation can choose it. `engine/faster_rcnn.py` runs it through the same steps as YOLOv2
on the shared data pipeline and evaluator: development training with validation AP50 every epoch, then the RPN NMS
IoU (proposal grid for Faster R-CNN: 0.50 / 0.70, final NMS fixed at 0.50) and operating confidence (0.2 / 0.3 /
0.4) chosen on validation and frozen, then one final test after retraining on train + validation.
`FasterRCNN_Project.ipynb` is built from the package like the YOLOv2 notebook. Person 2's branch imported an older
copy of Person 1's data pipeline; its augmentation settings, normalization and sampling weights are identical to
the package's. Differences from Person 2's run: the project seed (288275 instead of 42) and parallel data loading.

### Accuracy trial (run `frcnn_accuracy_20261104_1155`, Colab T4)

Same architecture; changes aimed at the proposal stage, where screwdrivers were lost, and at undertraining. They
were made together (no time for one-at-a-time runs), so their separate effects are not measured:

| Setting | Person 2's recipe | Accuracy trial | Reason |
|---|---|---|---|
| Epochs (early stopping) | 10 (none) | up to 60 (patience 10) | 10 was too few to see a plateau; 60 is the proposal's maximum |
| Learning rate | 5e-4, constant | 1e-3, 1-epoch warmup, cosine decay to 5% | low and unscheduled for SGD at batch 2; YOLOv2's schedule |
| Anchor ratios | 0.5, 1, 2 (15 anchors per cell) | 0.25, 0.5, 1, 2, 4 (25) | half the training screwdrivers are over 3:1 |
| RPN foreground IoU | 0.7 | 0.6 | thin objects rarely reach 0.7 with any anchor |
| Training proposals / sampled RoIs (max foreground) | 300 / 32 (8) | 600 / 128 (32) | standard is 128–512 RoIs; more examples for the RoI head |

Training: 358 images, 179 updates per epoch. Validation AP50 rose from 0.03 (epoch 1) to 0.49 (epoch 10), 0.60
(epoch 30) and a best of 0.701 at epoch 49; training stopped early after epoch 59. Training loss fell from 1.98
to 0.34.

| Validation (92 images) | Person 2's run | Accuracy trial | Target |
|---|---|---|---|
| **Overall AP50** | 0.266 | **0.701** | 0.70 |
| AP50:95 | | 0.318 | |
| Drill AP50 | | 0.953 | 0.75 |
| Scissor AP50 | | 0.817 | 0.65 |
| Screwdriver AP50 | 0.007 | 0.334 | 0.70 |
| Original / public AP50 | | 0.714 / 0.695 | |

Selected on validation: RPN NMS IoU 0.50 (AP50 0.701 against 0.696 at 0.70) and operating confidence 0.4. At
that confidence: precision 0.479, recall 0.774, F1 0.592 (TP 113, FP 123, FN 33). Model: 9,976,013 parameters,
40.0 MB checkpoint (YOLOv2: 37.3 M, 149 MB).

Where objects are lost (Person 2's diagnostic, validation):

| Class | Objects | Covered by a proposal (all / top 100) | RoI head correct on the exact box |
|---|---|---|---|
| Drill | 52 | 51 / 51 | 52 |
| Scissor | 47 | 45 / 40 | 40 |
| Screwdriver | 47 | 38 / 32 | 37 |

**Finding.** The changes fixed what they targeted: screwdrivers covered by the top 100 proposals rose from 2 to 32
of 47, and overall AP50 from 0.266 to 0.701, meeting the overall target. The gap is far larger than the seed and
loading differences could explain. Screwdrivers remain the weak class (0.334): 15 of 47 have no proposal in the
top 100, and the RoI head misnames 10 of 47 even given the exact box, which points at its coarse 5 × 5 pooled
features for thin objects.

**Decision.** The accuracy trial is the Faster R-CNN model for the final test (higher validation AP50 than
Person 2's recipe, the same selection rule as for YOLOv2). The final test has not been run yet: set
`RESUME_RUN = "frcnn_accuracy_20261104_1155"` and `RUN_FINAL_TEST = True` in the notebook.

### Comparison with YOLOv2's screwdriver work

YOLOv2's screwdriver gains came from finer spatial resolution (40 × 40 grid) plus thin anchor shapes (trial 5b:
0.434 → 0.605); making the detector more confident about small objects (trials 5c, 5d) raised screwdrivers but
not the overall score. The Faster R-CNN trial already uses stride-16 features and thin anchors. Its remaining
gaps suggest, if there is time for one more validation run: an anchor size of 16 px (many validation
screwdrivers are narrower than the smallest 32-px anchor) and RoIAlign 7 × 7 instead of 5 × 5. A stride-8
feature map (closer to YOLOv2's fix) is a larger change. As with YOLOv2, a further trial is kept only if it beats
the overall validation AP50 (0.701).

### Faster R-CNN limitations

- **Screwdrivers** are far below target on validation (0.334 vs 0.70), worse than YOLOv2 (0.605).
- **Many false positives at the operating point:** precision 0.479 at confidence 0.4. F1 was still rising at the
  top of the proposal's confidence grid (0.410, 0.513, 0.592), the same ceiling YOLOv2's trial 3 hit; AP50 is not
  affected, but precision and F1 at the operating point are.
- **Bundled changes:** the five changes of the accuracy trial were tested together, against a baseline measured
  on Person 2's branch (different seed), not rerun on this pipeline.
- **Loose boxes:** AP50:95 0.318 and mean matched IoU 0.73.
- **Software check** ran with the default (Person 2's) settings, so it confirms the wiring, not the trial's
  25-anchor model; the trial's own training run did use 25 anchors.
- **Slower training:** each image passes through the network on its own, so an epoch takes longer than YOLOv2's.

## Changes, in order

**3 Nov – integration onto one branch.** Person 1's full augmentation made the default (main had used only
flip/brightness/contrast); dataset path `./dataset`; duplicated code in the data module merged; PyTorch 2.2
compatibility for Intel Macs; dropped READMEs, GUI tests and Person 3's notebook restored. The data branch's
`dataset.zip` lacked `contract/`, which the software check needs; a complete zip was built and later committed.

**4 Nov – Colab speed.** The T4 was slower than a CPU runtime: data loading had no worker processes, and the
loss assignment and NMS looped box by box on the GPU. Added loader workers and moved those loops to the CPU;
results unchanged.

**4 Nov – Person 3 round 2.2.** Positive-objectness weight 1 → 5 integrated (his 8-image fitting check reached
0.95 AP50 on the 8 training images it was trained on; not a validation result).

**4 Nov – training recipe (round 3).** Options for warmup + cosine schedule, weight EMA, objectness prior,
mosaic, original-photo weighting and the proposal's random search; ImageNet backbone initialization (allowed by
the professor); 60 epochs / patience 10 (proposal section 4.2).

**4 Nov – one project.** Everything restructured into the `workbench_detection` package (config, boxes, data,
models, engine, evaluation, gui, visualize) with one command-line entry point, and `YOLO_Project.ipynb`
generated from it as a narrative notebook. Verified identical to the previous code on the same inputs: 18
component checks, a full development + validation run, and a final retrain + test. The notebook is resumable
after a Colab disconnect and never repeats the final test.

**4 Nov – finer grid option.** `stride: 16` setting (default 32 reproduces trial 3, verified by the 18 checks);
tested locally: 40 × 40 output, training and validation, checkpoint reload, inference on photos, correct box
positions, and the full trial 5 notebook.

**4 Nov – screwdriver options.** `anchors` and `objectness_target` settings added (defaults reproduce trial 3,
verified by the same 18 checks); used in trial 4. A stopped final test now restarts cleanly, while a finished
one is never repeated (tested locally).

**4 Nov – two anchors per object.** `anchors_per_object` setting (default 1, YOLOv2's single best-matching anchor,
reproduces every earlier run; checkpoints without it load with 1); used in trial 5d. The notebook's loss
demonstration confirmed twice the assigned anchors with 2. The assignment figure still draws only the best anchor.

**4 Nov – trial 5b as the recipe.** Trial 5b made the default (`TrainConfig`, `configs/yolov2.yaml`); checkpoints
saved before the newer settings existed load with the values they were trained with. `YOLO_Project.ipynb` rebuilt
with run `finegrid2_20261104_0654`'s outputs: development training, software check and validation from its first
session, the final test from its second.

**4 Nov – Faster R-CNN integrated.** Person 2's model from `RCNN_debug` moved into the package
(`models/faster_rcnn.py`, `engine/faster_rcnn.py`) with `FasterRCNN_Project.ipynb` and its builder; the notebook
builders now share `tools/notebook_parts.py`, and `lr_factor` has its own section so both use the schedule. New
`FasterRCNNConfig` settings (anchor ratios, RPN foreground IoU, proposal and RoI sampling counts, learning-rate
schedule) default to Person 2's values; the anchor ratios are saved in the checkpoint. The accuracy trial is the
notebook's default trial.

**4 Nov – GUI repository and model downloads.** The GUI, with only the code it needs for inference, is in its own
public repository (github.com/kemilii/EE3703-Group05-GUI), with the final YOLOv2 model attached to its `models`
release and a download script. The same model is attached to this repository's `models` release. Run folders that
were moved or downloaded now find their own checkpoint (the protocol records the Colab path).
