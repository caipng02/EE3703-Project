EE3703 Project - Group 05: Workbench Tool Detection
====================================================

Detects and localizes drills, scissors and screwdrivers in workbench photos,
comparing a two-stage Faster R-CNN with a one-stage YOLOv2, both with a
ResNet-18 backbone.

DRAFT: the Faster R-CNN notebook and the findings summary are still to come.


Project structure
-----------------
YOLO_Project.ipynb         The complete YOLOv2 system, self-contained and run
                           with outputs saved: data and class balance,
                           letterboxing and augmentation, class-aware
                           sampling, the ResNet-18 + YOLOv2 model and anchors,
                           the loss, evaluation protocol, training, validation,
                           the final test, error analysis and inference
FasterRCNN_Project.ipynb   The Faster R-CNN system (in progress)
Demo_Video.mp4             Inference on the 50 original test images by both
                           detectors, followed by the GUI demonstration
README.txt                 This file

GUI (bonus challenge) source code, in its own repository:
  https://github.com/kemilii/EE3703-Group05-GUI
  Upload a photo, choose YOLO or Faster R-CNN, and see the boxes, class
  labels and confidences. Setup and run instructions are in its README.

The notebooks are generated from the group's code repository, where the same
code is organized as the Python package workbench_detection (data, models,
engine, evaluation, gui) with a command-line interface and the browser GUI.

Dataset: 500 images at 640 x 640 (public and self-collected), split into
358 train / 92 validation / 50 held-out original test images.
Classes: 0 = drill, 1 = scissor, 2 = screwdriver.

Implementation: the YOLOv2 detection pipeline (ResNet-18 layers, head and
passthrough layer, k-means anchors, anchor assignment, box encoding and
decoding, loss, confidence filtering, class-wise NMS), the data pipeline,
training loop and evaluation logic are the group's own code. ImageNet weights
initialize the backbone (torchvision); AP is computed with pycocotools.


Dependencies
------------
Python 3.10 or newer
PyTorch and torchvision (CPU or CUDA build)
numpy, Pillow, opencv-python-headless, pycocotools, pyyaml, matplotlib


Summary of findings
-------------------
TODO - fill in after the final test-set evaluation of both detectors:
  - AP50 and AP50:95 per class and overall, for each detector
  - Inference speed for each detector
  - Typical successful detections and failure cases
