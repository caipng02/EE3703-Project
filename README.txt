EE3703 Project - Group 05: Workbench Tool Detection
====================================================

Detects and localizes drills, scissors and screwdrivers in 640 x 640
workbench photos, comparing a two-stage Faster R-CNN with a one-stage
YOLOv2, both on a ResNet-18 backbone.


Project structure
-----------------
FasterRCNN_Project.ipynb   The selected Faster R-CNN, self-contained and run
                           with outputs saved: data and class balance,
                           letterboxing and augmentation, class-aware
                           sampling, the model (stride-8 features, 55 anchors
                           per cell, RPN, RoI head), losses and decoding, the
                           evaluation protocol, training log and curves,
                           validation, the final test, error analysis and
                           inference on photos
YOLO_Project.ipynb         The selected YOLOv2, with the same sections: the
                           ResNet-18 + YOLOv2 model on a 40 x 40 grid with 9
                           k-means anchors, the YOLOv2 loss, training,
                           validation, the final test, error analysis and
                           inference
Demo_Video.mp4             Inference on the 50 original test images by both
                           detectors, followed by the GUI demonstration
README.txt                 This file

Both notebooks follow one protocol: settings are chosen on the 92 validation
images, then the model is retrained on train + validation (450 images) and
evaluated once on the 50 original test photos. Each notebook was run on a
Colab T4 GPU; training and validation were loaded from the saved run, and the
final retrain and test were run in the notebook.

GUI (bonus challenge) source code, in its own repository:
  https://github.com/kemilii/EE3703-Group05-GUI
  Upload a photo, choose YOLO or Faster R-CNN, and see the boxes, class
  labels and confidences. Setup and run instructions are in its README.
  The GUI uses final NMS IoU 0.30 for Faster R-CNN (the notebook's test
  results use 0.50); see the report, Section 7.3.

Dataset: 500 images at 640 x 640 (public and self-collected), split into
358 train / 92 validation / 50 held-out original test images.
Classes: drill, scissor, screwdriver.

Implementation boundary (details in the first cell of each notebook): the
data pipeline, both detection pipelines (Faster R-CNN RPN and RoI head,
anchor labelling, box coding, losses, proposal filtering and RoI sampling;
YOLOv2 ResNet-18 layers, head, anchors, assignment, loss and NMS), the
training loops and the evaluation logic are the group's own code. From
libraries: ImageNet ResNet-18 weights, and for Faster R-CNN the anchor
generator, RoIAlign and NMS operators (torchvision); AP50 and AP50:95
(pycocotools); layers, optimizers and autograd (PyTorch).


Dependencies
------------
Python 3.10 or newer
PyTorch and torchvision (CPU or CUDA build)
numpy, Pillow, opencv-python-headless, pycocotools, pyyaml, matplotlib


Summary of findings
-------------------
Test set: 50 original photos, 90 objects (26 drills, 36 scissors,
28 screwdrivers). AP50 / AP50:95:

                 YOLOv2           Faster R-CNN     Target (AP50)
  Overall        0.933 / 0.447    0.912 / 0.419    0.70
  Drill          0.906 / 0.448    0.856 / 0.413    0.75
  Scissor        0.943 / 0.462    0.951 / 0.449    0.65
  Screwdriver    0.950 / 0.431    0.929 / 0.394    0.70

Speed (batch 1, T4, including preprocessing and NMS): YOLOv2 41.9 ms per
image (23.9 FPS); Faster R-CNN 40.5 ms per image (24.7 FPS).
Model size: YOLOv2 37.3 M parameters (149 MB); Faster R-CNN 18.0 M
parameters (72 MB).

- Screwdrivers were the hard class for both detectors: thin, often diagonal
  and small. On validation, finer features (YOLOv2's 40 x 40 grid, Faster
  R-CNN's stride-8 map) and anchors that fit thin shapes gave the largest
  gains (YOLOv2 screwdriver AP50 0.43 -> 0.61; Faster R-CNN 0.33 -> 0.57).
- Faster R-CNN's main error type was partial boxes: the right class but a
  box covering only part of the object. Box offsets coded as in torchvision
  and partial boxes used as background examples reduced these and tightened
  the boxes (validation AP50:95 0.35 -> 0.38).
- Validation scores were lower than test scores for both detectors: 64% of
  the validation screwdrivers come from one cluttered public source, while
  the test holds only the group's own photos.
- Both detectors beat every target on the test. YOLOv2 is slightly more
  accurate overall; Faster R-CNN is better on scissors, has higher recall
  (0.92 against 0.89) and half the parameters, at the same speed.
- YOLOv2 produced far fewer false positives at its operating point (test
  precision 0.89 against 0.50 for Faster R-CNN), because its
  confidence is trained to match box overlap, whereas Faster R-CNN scores
  partial boxes of a correct class highly.
