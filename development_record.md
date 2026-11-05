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
not the overall score. The Faster R-CNN trial already uses stride-16 features and thin anchors.

### Next trials: anchor sizes, 7 × 7 pooling, stride 8 (5 Nov)

The project goal is the highest validation AP50 that can be reached, not only the targets. Before choosing changes,
the best IoU that any anchor reaches with each ground-truth box was computed for several anchor sets (no training;
torchvision's anchor generator on the 640 × 640 grid):

| Validation objects with an anchor at IoU ≥ 0.6 | Screwdriver (47) | Scissor (47) | Drill (52) |
|---|---|---|---|
| Accuracy trial: stride 16, sizes 32–512 (×2 steps), 5 ratios | 30 | 37 | 38 |
| + a 16 px size | 30 | 37 | 38 |
| Stride 16, sizes 16–512 in half-octave (×1.41) steps | 43 | 47 | 52 |
| Stride 8, half-octave sizes | 46 | 47 | 52 |

The 16 px anchor suggested earlier does nothing: only 5 of 47 validation screwdrivers are narrower than 32 px
(median 52 × 109 px). The gap is between sizes: a box halfway between two ×2 sizes fits neither well, for all three
classes. Each object's best anchor is already always RPN foreground, so the issue is how many good anchors teach
the RPN, not whether any does. Two trials, run on top of the accuracy trial:

| Trial | Change | Reason |
|---|---|---|
| `anchors` | sizes 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512 (55 anchors per cell); RoIAlign 7 × 7 | anchor table above; the RoI head misnamed 10 of 47 screwdrivers on the exact box from 5 × 5 features |
| `stride8` | `anchors` + layer 3 upsampled and added to layer 2 (1 × 1 lateral, 3 × 3 smoothing; one FPN level): an 80 × 80 map | YOLOv2's biggest screwdriver gain came from a finer grid |

New settings: `features` (`c4`, `c3c4`, `c5`) and `roi_pool`; checkpoints store both, and older checkpoints load
with Person 2's pairing (verified: the default and accuracy configurations give identical features, anchors,
losses and weights after a training step to the code before the change). On CPU, `stride8` takes about 1.5 times
as long per image as the accuracy trial; the RoI head grows from 6.6 M to 12.9 M parameters with 7 × 7 pooling.

As with YOLOv2, a trial is kept only if it beats the overall validation AP50 (0.701).

### Trial `stride8` (run `frcnn_stride8_20261105_0852`, Colab T4)

Training: validation AP50 0.04 (epoch 1), 0.44 (epoch 8), 0.61 (epoch 19), 0.73 (epoch 37) and a best of 0.769
at epoch 49; early stopping after epoch 59. Training loss 1.95 → 0.52. The curve is noisy: over the last 20
epochs validation AP50 moved between 0.68 and 0.77 from one epoch to the next (92 images), so the best epoch's
score is somewhat optimistic. The `anchors` trial was not run.

| Validation (92 images) | Accuracy trial | `stride8` | Change | Target |
|---|---|---|---|---|
| **Overall AP50** | 0.701 | **0.769** | +0.068 | 0.70 |
| AP50:95 | 0.318 | **0.350** | +0.032 | |
| Drill AP50 | 0.953 | **0.972** | +0.019 | 0.75 |
| Scissor AP50 | 0.817 | **0.832** | +0.015 | 0.65 |
| Screwdriver AP50 | 0.334 | **0.505** | +0.171 | 0.70 |
| Precision / recall (confidence 0.4) | 0.479 / 0.774 | **0.527** / **0.808** | | |
| F1 | 0.592 | **0.638** | +0.046 | |
| Original / public AP50 | 0.714 / 0.695 | **0.811** / **0.760** | | |
| Parameters / checkpoint | 10.0 M / 40 MB | 16.9 M / 68 MB | | |

Selected on validation: RPN NMS IoU 0.70 (AP50 0.769 against 0.709 at 0.50) and operating confidence 0.4 (F1:
0.2 → 0.536, 0.3 → 0.595, 0.4 → 0.638). TP 118, FP 106, FN 28; mean matched IoU 0.727. Screwdriver precision
0.378, recall 0.596.

Where objects are lost (validation, at the selected RPN NMS IoU):

| Class | Objects | Covered by a proposal (all / top 100) | RoI head correct on the exact box |
|---|---|---|---|
| Drill | 52 | 52 / 52 | 51 |
| Scissor | 47 | 38 / 38 | 40 |
| Screwdriver | 47 | 30 / 26 | 39 |

**Finding.** Every class improved, screwdrivers most (0.334 → 0.505), and the RoI head names more screwdrivers
correctly on the exact box (37 → 39 of 47). But fewer screwdrivers have a proposal at IoU ≥ 0.5 (38 → 30 of 47)
although far more anchors fit them; the AP gain comes from the RoI head's box refinement and classification, not
from better proposals. A likely cause (not yet tested): before RPN NMS only the 10 × 600 = 6,000 highest-scoring
anchors are kept, which was 15% of the 40,000 anchors at stride 16 but is 1.7% of the 352,000 at stride 8, so
lower-scoring screwdriver anchors are cut before NMS. The pre-NMS count is an inference setting and can be chosen
on validation without retraining.

**Decision.** `stride8` replaces the accuracy trial as the Faster R-CNN model for the final test (0.769 against
0.701). It is still below YOLOv2's trial 5b on validation (0.803; screwdriver 0.605). The final test has not been
run yet: set `RESUME_RUN = "frcnn_stride8_20261105_0852"` and `RUN_FINAL_TEST = True`. (The notebook's first
cells were later re-run without `RESUME_RUN`, which created an extra Drive folder `frcnn_stride8_20261105_0926`
holding only a software check; it is not used.)

### Screwdriver experiment (`experiments/FasterRCNN_screwdriver_experiment.ipynb`)

Goal before the final test: screwdriver validation AP50 of at least 0.70 (`stride8`: 0.505). Two steps:

1. **Re-validate `stride8` without retraining** (`SOURCE_RUN = "frcnn_stride8_20261105_0852"`, new run folder
   `frcnn_stride8_20261105_0852_revalidated`). Validation now also chooses how many top-scoring anchors enter RPN
   NMS: 6,000 (the old fixed value), 12,000, 24,000 or 48,000, each at RPN NMS IoU 0.50 and 0.70; the count with
   the best AP50 is frozen in the protocol with the other settings. Tests the pre-NMS explanation above.
2. **Train `stride8_v2`** (and `stride8_v2_mosaic` in a second session), aimed at the RoI head, where screwdrivers
   lose most (precision 0.378):

| Setting | `stride8` | `stride8_v2` | Reason |
|---|---|---|---|
| RoI head | one fully connected layer of 1024 | two (torchvision's head) | more capacity to tell screwdrivers from look-alike background |
| Anchors before RPN NMS in training | 6,000 | 24,000 | weakly scored screwdriver proposals reach the RoI head in training |
| Weight averaging | none | EMA, decay 0.997 (about 2 epochs) | validation AP50 swung by up to 0.09 between epochs |
| Mosaic (`stride8_v2_mosaic` only) | 0 | 0.5 | YOLOv2's recipe; more, smaller objects per image |

The three `stride8_v2` changes are bundled (one run each is several hours). A trial is kept only if it beats the
best overall validation AP50; with 47 validation screwdrivers, one screwdriver moves its AP50 by about 0.02, and
every further choice made on validation makes the validation score more optimistic, so the final test is the
only unbiased measure.

**Trial `stride8_v2` (run `frcnn_stride8_v2_20261105_1217`, Colab T4).** Validation AP50 during training (6,000
anchors before NMS, RPN NMS 0.70): 0.40 (epoch 8), 0.60 (epoch 17), 0.73 (epoch 37), best 0.752 at epoch 47;
early stopping after epoch 57. With EMA the curve is smooth: over the last 20 epochs it stayed between 0.72 and
0.75 (`stride8`: 0.68 to 0.77).

| Pre-NMS anchors | AP50 at RPN NMS 0.50 / 0.70 | Screwdriver AP50 at 0.50 / 0.70 |
|---|---|---|
| 6,000 (old fixed value) | 0.716 / 0.752 | 0.435 / 0.497 |
| 12,000 | 0.729 / 0.771 | 0.470 / 0.542 |
| **24,000 (selected)** | 0.753 / **0.780** | 0.517 / **0.554** |
| 48,000 | 0.758 / 0.780 | 0.530 / 0.554 |

| Validation (92 images) | `stride8` (6,000 anchors) | `stride8_v2` (24,000) |
|---|---|---|
| **Overall AP50** | 0.769 | **0.780** |
| AP50:95 | 0.350 | 0.350 |
| Drill / scissor / screwdriver AP50 | **0.972** / 0.832 / 0.505 | 0.949 / **0.838** / **0.554** |
| Precision / recall (confidence 0.4) | **0.527** / 0.808 | 0.364 / **0.836** |
| F1 | **0.638** | 0.507 |
| TP / FP / FN | 118 / **106** / 28 | **122** / 213 / **24** |
| Screwdriver precision / recall | **0.378** / 0.596 | 0.228 / **0.660** |
| Original / public AP50 | **0.811** / 0.760 | 0.809 / **0.771** |
| Screwdrivers covered by a proposal (all / top 100) | 30 / 26 | **36** / 26 |
| Screwdrivers named correctly on the exact box | 39 | **40** |

**Findings.** (1) The pre-NMS explanation holds: with the same model, going from 6,000 to 24,000 anchors raises
overall AP50 from 0.752 to 0.780 and screwdriver AP50 from 0.497 to 0.554, and screwdriver proposal coverage
rises from 30 to 36 of 47. (2) At the old 6,000 anchors, `stride8_v2` (0.752) is *below* `stride8` (0.769); its
higher headline score comes from the pre-NMS choice, which `stride8` has not been given yet. The two-layer head,
training pre-NMS count and EMA together did not raise AP50 on equal terms; EMA did remove the validation noise.
(3) False positives doubled (106 → 213) and precision fell to 0.364; F1 was again still rising at the top of the
confidence grid (0.396, 0.453, 0.507), so the operating point is limited by the proposal's grid, not chosen freely.
(4) Screwdrivers are still far from 0.70.

**`stride8` re-validated** (run `frcnn_stride8_20261105_0852_revalidated`, no training; notebook saved as
`experiments/FasterRCNN_screwdriver_experiment_run2.ipynb`, the `stride8_v2` run as `_run1`):

| Validation, each model's own selection | `stride8` | `stride8_v2` |
|---|---|---|
| Pre-NMS anchors, AP50 at RPN NMS 0.70: 6,000 / 12,000 / 24,000 / 48,000 | 0.769 / 0.775 / 0.780 / 0.780 | 0.752 / 0.771 / 0.780 / 0.780 |
| **Overall AP50** (24,000 selected for both) | 0.7796 | **0.7802** |
| AP50:95 | **0.358** | 0.350 |
| Drill / scissor / screwdriver AP50 | **0.956** / **0.861** / 0.522 | 0.949 / 0.838 / **0.554** |
| Precision / recall / F1 (confidence 0.4) | 0.330 / **0.849** / 0.475 | **0.364** / 0.836 / **0.507** |
| TP / FP / FN | **124** / 252 / **22** | 122 / **213** / 24 |
| Original / public AP50 | **0.855** / 0.759 | 0.809 / **0.771** |

The pre-NMS choice gives `stride8` +0.010 (`stride8_v2`: +0.028), and both end at the same overall AP50
(difference 0.0006, under one detection). With more candidates precision falls for both (`stride8`: 0.527 →
0.330), because more partial boxes survive to the output. By the selection rule `stride8_v2` stays ahead, and it
is the more reliable of the two (smooth EMA curve rather than a best epoch picked from swings of up to 0.09).

**Error review of `stride8_v2`** (validation, confidence 0.4): of 213 false positives, 125 are *localization*
errors (the right class, IoU 0.1–0.5 with the object), 63 background, 24 duplicates and 1 wrong class; 24
objects are missed. The examples show one drill covered by several high-confidence partial boxes that overlap each
other too little for the final NMS. Two causes in the code:

1. **Box offsets coded without scaling.** The RoI head regressed raw offsets with smooth L1 at beta 1 (weight 10),
   so an error of a few percent of the box size gave almost no gradient; torchvision's Faster R-CNN scales the
   offsets by (10, 10, 5, 5) and uses beta 1/9. Matches the loose boxes (AP50:95 0.35, matched IoU 0.72) and thin
   screwdrivers falling below IoU 0.5.
2. **Background RoIs sampled uniformly**, mostly empty background, so partial boxes are rarely trained as
   background. Fast R-CNN draws background RoIs from IoU 0.1–0.5. (With an untrained RPN, uniform sampling gave
   about 40% partial boxes; `roi_hard_negatives = 0.75` gives 75%. With a trained RPN the uniform share may be
   higher, so this fix may matter less than the first.)

**Trial `stride8_v3`** (run `frcnn_stride8_v3_20261105_1304`, Colab T4,
`experiments/FasterRCNN_boxfix_experiment.ipynb`): `stride8_v2` with `roi_box_coder = "standard"` and
`roi_hard_negatives = 0.75`. The notebook's outputs were lost (the Colab session re-ran it); the results below come
from the run folder's `report/validation/result.json`. The training curve and best epoch are not recorded here.

| Pre-NMS anchors | AP50 at RPN NMS 0.50 / 0.70 |
|---|---|
| 6,000 | 0.737 / 0.773 |
| 12,000 | 0.754 / 0.782 |
| **24,000 (selected)** | 0.776 / **0.788** |
| 48,000 | 0.778 / 0.787 |

| Validation (92 images), each model's own selection | `stride8_v2` | **`stride8_v3`** | Change |
|---|---|---|---|
| **Overall AP50** | 0.780 | **0.788** | +0.008 |
| AP50:95 | 0.350 | **0.394** | +0.044 |
| Mean matched IoU | 0.719 | **0.746** | +0.027 |
| Drill AP50 | **0.949** | 0.936 | −0.013 |
| Scissor AP50 | 0.838 | **0.854** | +0.016 |
| Screwdriver AP50 | 0.554 | **0.574** | +0.020 |
| Screwdriver AP50:95 | 0.227 | **0.253** | +0.026 |
| Precision / recall (confidence 0.4) | 0.364 / 0.836 | **0.412** / 0.836 | |
| F1 | 0.507 | **0.552** | +0.045 |
| TP / FP / FN | 122 / 213 / 24 | 122 / **174** / 24 | |
| False positives: localization / background / duplicate / wrong class | 125 / 63 / 24 / 1 | **103** / **58** / **12** / 1 | |
| Screwdriver precision / recall | 0.228 / 0.660 | **0.263** / 0.660 | |
| Original / public AP50 | 0.809 / 0.771 | **0.828** / 0.769 | |

Settings frozen on validation: 24,000 anchors before RPN NMS, RPN NMS IoU 0.70, final NMS IoU 0.50, operating
confidence 0.4. Model: 17,978,723 parameters, 72.0 MB checkpoint.

**Finding.** The fixes acted where the error review pointed: boxes are tighter (AP50:95 +0.044, matched IoU 0.719
→ 0.746), and localization and duplicate false positives fell (125 → 103, 24 → 12). Recall is unchanged (same
24 objects missed), so the gain comes from fewer and better-placed boxes, not from finding more objects. The
overall AP50 gain (+0.008) is about one detection's worth on this validation set; the AP50:95 and error-count
changes are larger and consistent with it. Screwdrivers improved (0.554 → 0.574) but remain far from 0.70, and
precision at the operating point is still low (0.412; `stride8` before the pre-NMS choice: 0.527).

**Decision.** `stride8_v3` is the Faster R-CNN model for the final test (highest overall validation AP50: 0.788
against 0.780). It is still below YOLOv2's trial 5b on validation (0.803; screwdriver 0.605). The final test has not
been run: in the box-fix notebook set `RESUME_RUN = "frcnn_stride8_v3_20261105_1304"` and `RUN_FINAL_TEST = True`.

### Faster R-CNN limitations

- **Screwdrivers** are below target on validation (`stride8_v3`: 0.574 vs 0.70), worse than YOLOv2 (0.605).
- **Proposals miss screwdrivers:** in `stride8_v2`, 11 of 47 validation screwdrivers have no proposal at IoU ≥ 0.5
  even with 24,000 anchors before NMS (`stride8` with 6,000: 17); not measured for `stride8_v3`.
- **Many false positives at the operating point:** precision 0.412 at confidence 0.4 (`stride8_v3`; 174 false
  positives, 103 of them partial boxes). Choosing more anchors before NMS raised AP50 but lowered precision for
  every model (`stride8`: 0.527 → 0.330). F1 was still rising at the top of the proposal's confidence grid in every
  run, the same ceiling YOLOv2's trial 3 hit; AP50 is not affected, but precision and F1 at the operating point are.
- **Noisy validation curve (`stride8`):** epoch-to-epoch swings of up to 0.09 AP50 near the end of training; EMA
  (`stride8_v2`, `stride8_v3`) removes most of it.
- **Many choices made on validation:** anchors, stride, pre-NMS count, box coding and the trial itself were all
  selected on the same 92 images (47 screwdrivers), so the validation scores are optimistic; the final test is the
  only unbiased measure.
- **Lost notebook outputs (`stride8_v3`):** the run's notebook was re-run by Colab and its outputs lost; the results
  come from the run folder's saved files, and the training curve is not in this record.
- **Bundled changes:** the five changes of the accuracy trial were tested together, against a baseline measured
  on Person 2's branch (different seed), not rerun on this pipeline.
- **Loose boxes:** AP50:95 0.394 and mean matched IoU 0.75 (`stride8_v3`; `stride8`: 0.350 and 0.73).
- **Software check** ran with the default (Person 2's) settings, so it confirms the wiring, not the trial's
  stride-8 model; the trial's own training run did use it.
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

**5 Nov – Faster R-CNN feature and anchor options.** `features` (`c4` stride 16, `c3c4` stride 8, `c5` stride 32)
and `roi_pool` settings and a `half_octave` anchor profile; the stride no longer follows from the anchor sizes
(before, any sizes other than the dense profile silently switched to stride 32 with 7 × 7 pooling). Trials
`anchors` and `stride8` added to `FasterRCNN_Project.ipynb`, whose default trial is now `stride8` (its saved
outputs are run `frcnn_stride8_20261105_0852`'s). Defaults reproduce the earlier code exactly (same features, anchors,
losses and weights after one update); old checkpoints load unchanged.

**5 Nov – screwdriver experiment options.** Validation chooses the number of anchors kept before RPN NMS
(`PRE_NMS_GRID`) and freezes it in `evaluation_protocol.json` (`rpn_pre_nms`; protocols without it keep the fixed
6,000). New settings `roi_fc_layers`, `train_pre_nms` and `ema` (defaults reproduce the last commit exactly: same
features, anchors, losses and weights after one update). `ModelEMA` moved to the trainer's schedule section so the
Faster R-CNN notebook includes it. `tools/build_faster_rcnn_notebook.py --experiment` writes
`experiments/FasterRCNN_screwdriver_experiment.ipynb`, which can re-validate a finished run (`SOURCE_RUN`) without
retraining. `FasterRCNN_Project.ipynb` was not rebuilt and keeps the `stride8` run's code and outputs.

**5 Nov – RoI box coding and hard negatives.** Settings `roi_box_coder` (`plain`, Person 2's; `standard`,
torchvision's offset scaling and smooth-L1 beta, `BOX_CODERS`), saved in checkpoints so decoding matches training,
and `roi_hard_negatives`. Defaults reproduce the last commit exactly (same features, anchors, losses and weights
after one update); offsets decode back to the exact box with both codings. `--boxfix` builds
`experiments/FasterRCNN_boxfix_experiment.ipynb` with trial `stride8_v3`.

**5 Nov – submission notebooks (project briefing).** The briefing asks for notebooks with the selected
architecture only ("do not include code for other ... architectures used only for comparison"), pre-run with
training logs, curves and inference results visible. Both packages were reduced to the selected models:
- Faster R-CNN: stride-8 features, half-octave anchors, RoIAlign 7 × 7, two fully connected layers and the
  torchvision box coding are now fixed in `models/faster_rcnn.py`; `FasterRCNNConfig`'s defaults are the
  `stride8_v3` recipe. Removed: stride-16/32 features, other anchor profiles, the one-layer head, the plain box
  coding and Person 2's unbalanced-loss path.
- YOLOv2: the 40 × 40 grid and 9 anchors are fixed; removed the stride-32 passthrough, the `objectness_target`
  and `anchors_per_object` options and the hyperparameter search (never used for the selected model).
- Checkpoints and run settings saved by the development code load only if they used the selected architecture.
  For both models the reduced code gives bit-identical features, anchors, losses, detections and weights after
  one update to the code before the change, on real training images; the final YOLOv2 checkpoint and a Faster
  R-CNN checkpoint saved by the old code load and detect.
- When a notebook resumes a finished run it now reprints the training log from the saved per-epoch history.
- `FasterRCNN_Project.ipynb` resumes run `frcnn_stride8_v3_20261105_1338` with `RUN_FINAL_TEST = True`;
  `YOLO_Project.ipynb` resumes `finegrid2_20261104_0654` (test already done, loaded). The previous notebooks with
  their outputs are kept as `experiments/FasterRCNN_stride8_run.ipynb` and `experiments/YOLO_finegrid2_run.ipynb`.
  The experiment notebooks in `experiments/` embed the code they ran with, including the removed options.
- YOLOv2 code clarity: descriptive names and step comments in the loss, decoding and training loop; docstrings
  for the remaining substantive functions (functions without one: 45 → 25, all private helpers or one-line
  methods).

