"""Run Person 3's YOLOv2 on Person 1's dataset and score it with Person 4's evaluator."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

import person4_evaluation as scoring
from workbench_data import ToolDataset, boxes_to_yolo, collate, make_train_sampler, normalize
from yolo_model import CLASSES, YOLOv2ResNet18, cluster_anchors, postprocess, seed_all, yolo_loss

MODEL_NAME = "YOLOv2"
SCORE_FLOOR = 0.001


def targets_for_loss(targets):
    return [torch.cat((row["labels"].float()[:, None], boxes_to_yolo(row["boxes"])), dim=1)
            for row in targets]


def infer_pixels(model, images, device, confidence, nms_iou):
    """Decode the detector's normalized output on the saved 640-pixel canvas."""
    raw = model(images.to(device)).float()
    return detections_from_raw(raw, model.anchors, confidence, nms_iou)


def detections_from_raw(raw, anchors, confidence, nms_iou):
    results = postprocess(raw, anchors, conf=confidence, nms_iou=nms_iou)
    for row in results:
        row["boxes"] *= 640
    return results


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("YOLO checkpoint class order differs from the dataset.")
    model = YOLOv2ResNet18(checkpoint["anchors"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def predict_uploaded_image(model, image, device, confidence, nms_iou, letterbox, restore_box):
    canvas, transform = letterbox(image)
    tensor = normalize(np.array(canvas))
    with torch.inference_mode():
        result = infer_pixels(model, tensor[None], device, confidence, nms_iou)[0]
    detections = []
    for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
        restored = restore_box(box, transform)
        if restored[2] > restored[0] and restored[3] > restored[1]:
            detections.append({"label": CLASSES[label], "confidence": score, "box": restored})
    return detections


def make_loader(dataset, batch, training=False):
    sampler = make_train_sampler(dataset) if training else None
    return DataLoader(dataset, batch_size=batch, sampler=sampler, collate_fn=collate)


def training_step(model, optimizer, images, targets, device, scaler):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        raw = model(images.to(device))
    loss, _ = yolo_loss(raw.float(), targets_for_loss(targets), model.anchors)
    if not torch.isfinite(loss):
        raise ValueError("YOLO training produced nonfinite loss.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def anchors_from_training(dataset):
    widths = torch.from_numpy(np.concatenate([boxes[:, 2:] - boxes[:, :2] for boxes in dataset.boxes]))
    return cluster_anchors(widths / 640)


@torch.inference_mode()
def collect_predictions(model, loader, gt, device, nms_candidates, checkpoint, floor, loss=False):
    model.eval()
    targets, collected = [], {iou: [] for iou in nms_candidates}
    total_loss = 0.0
    for images, rows in loader:
        raw = model(images.to(device)).float()
        if loss:
            batch_loss, _ = yolo_loss(raw, targets_for_loss(rows), model.anchors)
            total_loss += batch_loss.item() * len(rows)
        for iou in nms_candidates:
            collected[iou].extend(detections_from_raw(raw, model.anchors, floor, iou))
        targets.extend(rows)
    exports = [scoring.predictions_from_outputs(
        targets, collected[iou], gt, MODEL_NAME,
        label_mapping=dict(enumerate(range(len(CLASSES)))),
        metadata={"checkpoint": str(checkpoint), "model_family": "yolov2",
                  "nms_stage": "final", "nms_iou": iou, "nms_rule": "class_aware",
                  "export_score_floor": floor},
    ) for iou in nms_candidates]
    return exports, total_loss / len(targets) if loss else None


def train(args, device):
    seed_all()
    args.output.mkdir(parents=True, exist_ok=True)
    final = args.final_from is not None
    protocol = None
    train_set = ToolDataset(args.dataset_root, "trainval" if final else "train", augment=True)
    if final:
        source = torch.load(args.final_from, map_location="cpu", weights_only=True)
        protocol = scoring.load_json(args.protocol)
        policy = protocol["models"][MODEL_NAME]
        selected = Path(policy["validation_checkpoint"]).resolve()
        if selected != args.final_from.resolve():
            raise ValueError("Use the validation checkpoint recorded in the selected protocol.")
        if tuple(source["classes"]) != CLASSES or source.get("stage") != "validation":
            raise ValueError("Final retraining requires a compatible validation checkpoint.")
        anchors, epochs = source["anchors"], source["best_epoch"]
        batch, lr, weight_decay = (source["config"][key] for key in ("batch", "lr", "weight_decay"))
    else:
        if args.epochs is None:
            raise ValueError("Specify --epochs for a development training run.")
        anchors, epochs = anchors_from_training(train_set), args.epochs
        batch, lr, weight_decay = args.batch, args.lr, args.weight_decay
    loader = make_loader(train_set, batch, training=True)
    if not final:
        gt = scoring.load_person1_ground_truth(args.dataset_root, "valid")
        valid = make_loader(ToolDataset(args.dataset_root, "valid"), batch)
    model = YOLOv2ResNet18(anchors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = args.output / "best.pt"
    best, best_epoch, history = -1, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = sum(training_step(model, optimizer, images, targets, device, scaler) * len(targets)
                    for images, targets in loader)
        row = {"epoch": epoch, "train_loss": total / len(loader.dataset)}
        if not final:
            exports, validation_loss = collect_predictions(
                model, valid, gt, device, [0.50], checkpoint, SCORE_FLOOR, loss=True)
            aps, _ = scoring.coco_ap(gt, exports[0], scoring.EvaluationConfig(ap_score_floor=SCORE_FLOOR))
            score = sum(ap["0.50"] for ap in aps) / len(aps)
            row.update(val_loss=validation_loss, val_map50=score)
        history.append(row)
        print(" | ".join(f"{key}={value:.4f}" for key, value in row.items()), flush=True)
        if final or score > best:
            if not final:
                best = score
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "anchors": anchors, "classes": CLASSES,
                        "config": {"batch": batch, "lr": lr, "weight_decay": weight_decay},
                        "best_epoch": epoch, "stage": "final" if final else "validation",
                        "protocol": protocol}, checkpoint)
        if not final and epoch - best_epoch >= 10:
            break
    scoring.plot_training_history(history, args.results or args.output, MODEL_NAME)
    print(f"Checkpoint: {checkpoint.resolve()} (epoch {best_epoch})")


def inspect_integration(args, device):
    seed_all()
    gt = scoring.load_person1_ground_truth(args.dataset_root, "contract")
    names = {image["image_id"] for image in gt["images"]}
    training = ToolDataset(args.dataset_root, "train")
    subset = Subset(training, [i for i, name in enumerate(training.names) if name in names])
    loader = make_loader(subset, args.batch)
    images, targets = next(iter(loader))
    model = YOLOv2ResNet18(anchors_from_training(training)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss = training_step(model, optimizer, images, targets, device, scaler)
    print(f"Training batch {tuple(images.shape)} | one-update loss={loss:.4f}")
    print(f"First image {targets[0]['file_name']} | YOLO targets [class, cx, cy, w, h]:")
    print(targets_for_loss(targets)[0])
    predictions, _ = collect_predictions(model, loader, gt, device, [0.50], "inspection", SCORE_FLOOR)
    # Keep the example readable while the model is still near random initialization.
    for row in predictions[0]["images"]:
        for field in ("boxes", "labels", "scores"):
            row[field] = row[field][:5]
    print("Inspection displays up to five decoded detections per image.")
    first = predictions[0]["images"][0]
    print(f"{first['image_id']}: {len(first['boxes'])} detections; first pixel xyxy boxes={first['boxes'][:2]}")
    predictions[0]["metadata"]["prediction_kind"] = "software_fixture"
    scoring.run_evaluation(
        gt, predictions, args.output, scoring.EvaluationConfig(ap_backend="coco", ap_score_floor=SCORE_FLOOR,
                                                               operating_confidence=SCORE_FLOOR),
        split="contract", image_root=args.dataset_root / "contract" / "images", evaluation_purpose="software_check")


def evaluate(args, device):
    final = args.split == "test"
    protocol = scoring.load_json(args.protocol) if final else None
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if final and (checkpoint.get("stage") != "final" or checkpoint.get("protocol") != protocol):
        raise ValueError("Test requires the final checkpoint retrained with this frozen protocol.")
    if not final and checkpoint.get("stage") == "final":
        raise ValueError("Use the development checkpoint to select settings on validation.")
    floor = protocol["ap_score_floor"] if final else args.ap_score_floor
    candidates = [protocol["models"][MODEL_NAME]["nms"]["nms_iou"]] if final else scoring.PROJECT_NMS_GRIDS["yolov2"]
    gt = scoring.load_person1_ground_truth(args.dataset_root, args.split, allow_test=final)
    image_root = args.dataset_root / args.split / "images"
    loader = make_loader(ToolDataset(args.dataset_root, args.split, allow_test=final), args.batch)
    predictions, _ = collect_predictions(model, loader, gt, device, candidates, args.checkpoint.resolve(), floor)
    size = scoring.model_size_summary(model, args.checkpoint)
    for prediction in predictions:
        prediction["metadata"]["model_size"] = size
    if args.benchmark:
        # Decode before timing. Preprocessing and CPU output conversion happen inside.
        samples = []
        for image in gt["images"]:
            with Image.open(image_root / image["file_name"]) as opened:
                samples.append(np.array(opened.convert("RGB")))

        @torch.inference_mode()
        def predict(image):
            return infer_pixels(model, normalize(image)[None], device,
                                protocol["models"][MODEL_NAME]["operating_confidence"], candidates[0])

        # Benchmark the frozen final NMS setting.
        predictions[0]["metadata"]["benchmark"] = scoring.benchmark_inference(
            predict, samples, device=str(device), precision="float32",
            synchronize=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None)
    scoring.run_evaluation(
        gt, predictions, args.output, scoring.EvaluationConfig(ap_score_floor=floor),
        split=args.split, image_root=image_root, project_protocol=True,
        select_nms=not final, frozen_protocol=protocol, results_dir=args.results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inspect", "train", "evaluate"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Checkpoint, protocol, or inspection folder.")
    parser.add_argument("--results", type=Path, help="Report metrics and figures folder.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--final-from", type=Path)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--ap-score-floor", type=float, default=SCORE_FLOOR)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.batch < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("Batch size and epoch count must be positive.")
    if args.final_from and (args.mode != "train" or not args.protocol):
        parser.error("--final-from requires train mode and --protocol.")
    if args.mode == "evaluate" and (not args.checkpoint or (args.split == "test" and not args.protocol)):
        parser.error("Evaluation needs --checkpoint. Final test also needs --protocol.")
    if args.benchmark and (args.mode != "evaluate" or args.split != "test"):
        parser.error("--benchmark uses final test with frozen settings.")
    root = Path(__file__).resolve().parent
    stage = {"train": "final_training" if args.final_from else "training",
             "evaluate": "validation" if args.split == "valid" else "test",
             "inspect": "inspection"}[args.mode]
    args.output = args.output or root / "outputs" / "yolov2" / stage
    if args.mode == "inspect":
        if args.results:
            parser.error("Inspection artifacts belong in --output.")
    else:
        args.results = args.results or root / "results" / "yolov2" / stage
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    {"inspect": inspect_integration, "train": train, "evaluate": evaluate}[args.mode](args, device)


if __name__ == "__main__":
    main()
