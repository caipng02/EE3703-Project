# EE3703 Project: Workbench Tool Detection

## Project Definition

Develop an object-detection system that identifies and localizes **drills,
scissors, and screwdrivers** in workbench images. Each detected tool should be
shown with a bounding box, class label, and confidence score.

The project compares a two-stage **Faster R-CNN** detector with a one-stage
**YOLO** detector for inventory and safety assistance. The proposal selects
Faster R-CNN with a ResNet-18 backbone and YOLOv2 with a ResNet-18 backbone;
implementation details remain subject to group agreement.

The dataset combines public and self-collected images, with 640 × 640 model
inputs and 50 original images reserved for final testing. The comparison will
consider detection performance, inference speed, and successful and failed
predictions.

This repository is a shared space for uploading the group's code and bringing
the detector implementations together for evaluation.

## Task Splits

| Role | Responsibility |
|---|---|
| Person 1 | Dataset strategy, annotation, preprocessing, and data limitations |
| Person 2 | Faster R-CNN methodology, implementation, and results |
| Person 3 | YOLOv2 methodology, implementation, and results |
| Person 4 | Experimental design, evaluation metrics, comparisons, error analysis, and final consolidation |
| Persons 1 and 4 | Interactive demo GUI |

## Project Deliverables and Deadlines

| Deliverable | Deadline / Schedule | Weight |
|---|---|---|
| Project proposal | **6 October 2026 — Week 8** | 10% |
| Project report | **6 November 2026 — Week 12** | 10% |
| Demo video and code | **6 November 2026 — Week 12** | 10%, including any GUI bonus |
| In-class presentation | **Week 13 Tuesday class**; a Thursday session will be scheduled only if Tuesday is insufficient | 10% |

Deadlines and presentation arrangements follow the project briefing; it does
not specify a submission time.

### 1. Project Proposal

Submit **`Proposal_Project_Group_05.pdf`**, covering the problem and objectives,
dataset strategy, technical approach, evaluation plan, timeline, and task
allocation.

### 2. Project Report

Submit **`Report_Project_Group_05.pdf`**, covering the executed dataset strategy,
methodology and implementation, experiments and results, discussion, and error
analysis. Include visual examples of successful detections and failures, and
explain which components were implemented manually versus provided by APIs.

### 3. Demo Video and Code

Submit **`Demo_Project_Group_05.zip`**, containing **exactly four files**:

| File | Required content |
|---|---|
| `Demo_Video.mp4` | A three-minute demonstration of both selected detectors performing inference on the 50 original test images |
| `FasterRCNN_Project.ipynb` | Preprocessing, model configuration, training, and inference for the selected Faster R-CNN architecture |
| `YOLO_Project.ipynb` | Preprocessing, model configuration, training, and inference for the selected YOLO architecture |
| `README.txt` | Project structure, dependencies, and a brief summary of findings |

Run the notebooks before submission and save the relevant training logs, loss
curves, and inference outputs. The teaching team will inspect saved outputs
rather than run the code.

**Optional interactive GUI:** allow image upload, model selection, and display
of predictions with boxes, labels, and confidence scores. Append its recording
after the three-minute inference segment; the GUI segment is outside that time
limit. The GUI can earn up to 2% bonus, with the demo/code score capped at 10%.

The browser interface is in [`gui/`](gui/README.md), with upload, detector
selection and annotated results displayed on the page. See its README for startup and
instructions for connecting each trained detector. The model adapters are
disabled until the group's inference code and weights are available.

### 4. In-Class Presentation

Deliver a **12-minute group presentation followed by 5 minutes of Q&A**.
Every group member must present a section. Cover the problem and dataset,
architectures and implementation, results, visual evidence, and error analysis.
The class starts at **2:00 PM**, when the presentation order is drawn.
