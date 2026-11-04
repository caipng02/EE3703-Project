"""Train and evaluate a workbench detector."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LOCAL_DATASET = ROOT.parent / "tmp" / "person1_data_branch" / "model_training_handoff" / "dataset"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("yolov2", "faster-rcnn"), default="yolov2")
    parser.add_argument("--dataset-root", type=Path, default=LOCAL_DATASET)
    parser.add_argument("--output", type=Path,
                        help="Checkpoints, protocols and inspection files.")
    parser.add_argument("--results", type=Path,
                        help="Report metrics and figures.")
    parser.add_argument("--checkpoint", type=Path, help="Reuse a compatible development checkpoint and skip training.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10, help="Stop after this many epochs without higher validation AP50.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--device", help="Defaults to CUDA if available, otherwise CPU.")
    parser.add_argument("--ap-score-floor", type=float, default=0.001)
    parser.add_argument("--inspect", action="store_true", help="Run one training update and generate visible examples on the training sample.")
    parser.add_argument("--final-test", action="store_true", help="After validation, retrain with the selected settings and score the held-out test.")
    parser.add_argument("--gui", action="store_true", help="Serve the trained model and selected settings locally after evaluation.")
    args = parser.parse_args()
    args.output = args.output or ROOT / "outputs" / args.model
    args.results = args.results or ROOT / "results" / args.model
    if args.model == "faster-rcnn":
        if args.inspect or args.final_test or args.gui:
            parser.error("The Faster R-CNN entry point supports development validation and checkpoint evaluation.")
        from faster_rcnn_diagnostics import main as run_faster_rcnn

        command = ["checkpoint" if args.checkpoint else "reproduce",
                   "--dataset-root", str(args.dataset_root), "--output", str(args.output),
                   "--results", str(args.results), "--epochs", str(args.epochs),
                   "--batch", str(args.batch), "--patience", str(args.patience),
                   "--score-floor", str(args.ap_score_floor)]
        for flag, value in (("--device", args.device), ("--lr", args.lr),
                            ("--weight-decay", args.weight_decay), ("--checkpoint", args.checkpoint)):
            if value is not None:
                command.extend((flag, str(value)))
        run_faster_rcnn(command)
        return
    if args.lr is None:
        args.lr = 0.0001
    if args.weight_decay is None:
        args.weight_decay = 0.0001
    if args.patience < 1:
        parser.error("--patience must be at least 1.")
    if args.inspect and (args.checkpoint or args.final_test or args.gui):
        parser.error("--inspect runs independently of checkpoints, final testing, and the GUI.")
    if args.epochs < 1 or args.batch < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Use positive epochs, batch size, and learning rate, and nonnegative weight decay.")
    if not 0 <= args.ap_score_floor <= 0.20:
        parser.error("AP score floor must be between 0 and the lowest operating confidence (0.20).")
    if not (args.dataset_root / "manifest.csv").is_file():
        parser.error("Point --dataset-root to Person 1's unpacked dataset folder containing manifest.csv.")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"Checkpoint file not found: {args.checkpoint}")
    if args.inspect:
        directories = [args.output / "inspection"]
    else:
        stages = ["validation"]
        if args.checkpoint is None:
            stages.insert(0, "training")
        if args.final_test:
            stages.extend(("final_training", "test"))
        directories = [root / name for root in (args.output, args.results) for name in stages]
    if not args.inspect:
        work, report = args.output.resolve(), args.results.resolve()
        if work.is_relative_to(report) or report.is_relative_to(work):
            parser.error("--output and --results must be separate directories.")
    for directory in directories:
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            parser.error(f"Choose a fresh directory for this run: {directory}")

    import torch
    from integrate_yolo import evaluate, inspect_integration, train

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    print(f"Device: {device} | Dataset: {args.dataset_root.resolve()}", flush=True)

    def stage(name, **overrides):
        return argparse.Namespace(**{**vars(args), "output": args.output / name,
                                     "results": None if args.inspect else args.results / name, **overrides})

    if args.inspect:
        inspect_integration(stage("inspection"), device)
        return

    checkpoint = args.checkpoint
    if checkpoint is None:
        print("\nDevelopment training", flush=True)
        training = stage("training", final_from=None)
        train(training, device)
        checkpoint = training.output / "best.pt"

    print("\nValidation and threshold selection", flush=True)
    validation = stage("validation", split="valid", checkpoint=checkpoint, benchmark=False)
    evaluate(validation, device)
    protocol = validation.output / "evaluation_protocol.json"

    if args.final_test:
        print("\nFinal retraining with frozen validation settings", flush=True)
        final_training = stage("final_training", final_from=checkpoint, protocol=protocol)
        train(final_training, device)
        checkpoint = final_training.output / "best.pt"
        print("\nHeld-out evaluation and timing", flush=True)
        evaluate(stage("test", split="test", checkpoint=checkpoint, protocol=protocol, benchmark=True), device)

    print(f"\nReport metrics and figures: {args.results.resolve()}", flush=True)
    print(f"Checkpoints and protocol: {args.output.resolve()}", flush=True)
    if args.gui:
        env = os.environ.copy()
        env["WORKBENCH_YOLO_CHECKPOINT"] = str(checkpoint.resolve())
        env["WORKBENCH_YOLO_PROTOCOL"] = str(protocol.resolve())
        env["WORKBENCH_YOLO_DEVICE"] = str(device)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
        try:
            subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "gui" / "server.py")],
                           cwd=ROOT, env=env, check=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
