"""Command-line interface: python -m workbench_detection <command> --help

    inspect   software check on 8 training images
    train     development training (or --search N) and validation; --final-test also runs the test
    test      final retrain and the single evaluation on the 50 original test images
    predict   draw detections on your own photos
    gui       browser demo at http://127.0.0.1:8765
    score     score saved prediction JSON files (e.g. from another detector)
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

from .config import TrainConfig, load_config


def _device(name):
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def _parse_overrides(pairs):
    """--set key=value pairs, converted to the TrainConfig field types."""
    types = {field.name: field.type for field in fields(TrainConfig)}
    changes = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in types or not value:
            raise SystemExit(f"--set expects key=value with key in: {', '.join(types)}")
        kind = types[key]
        if kind in (bool, "bool"):
            if value.lower() not in ("true", "false"):
                raise SystemExit(f"--set {key} expects true or false")
            changes[key] = value.lower() == "true"
        else:
            changes[key] = {"int": int, "float": float, "str": str}.get(getattr(kind, "__name__", kind), str)(value)
    return changes


def _setup(device):
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    print(f"Device: {device}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m workbench_detection", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text):
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--device", help="cuda or cpu (default: cuda when available)")
        return sub

    sub = command("inspect", "Software check on the 8 contract images; the scores are not detector results.")
    sub.add_argument("--dataset", type=Path, default=Path("dataset"))
    sub.add_argument("--out", type=Path, default=Path("runs/inspect"))

    sub = command("train", "Development training (or a search) followed by validation.")
    sub.add_argument("--dataset", type=Path, default=Path("dataset"))
    sub.add_argument("--run", type=Path, required=True, help="New run folder, e.g. runs/pretrained_v1")
    sub.add_argument("--config", type=Path, default=Path("configs/yolov2.yaml"))
    sub.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override settings, e.g. lr=3e-4 ema=false")
    sub.add_argument("--search", type=int, default=0, help="Random search over N of the 12 lr x weight decay x batch combinations")
    sub.add_argument("--final-test", action="store_true", help="Then retrain and evaluate once on the test split")

    sub = command("test", "Final retrain on train + validation and the single evaluation on the test split.")
    sub.add_argument("--dataset", type=Path, default=Path("dataset"))
    sub.add_argument("--run", type=Path, required=True)

    sub = command("predict", "Draw detections on photos with a run's best checkpoint.")
    sub.add_argument("--run", type=Path, required=True)
    sub.add_argument("images", nargs="+", type=Path)
    sub.add_argument("--out", type=Path, help="Output folder (default: <run>/predictions)")

    sub = command("gui", "Browser demo: upload a photo, choose a model, see the detections.")
    sub.add_argument("--run", type=Path, help="Run whose best YOLOv2 checkpoint to serve")
    sub.add_argument("--port", type=int, default=8765)

    commands.add_parser("score", help="Score saved prediction JSON files (see score --help).", add_help=False)

    args, rest = parser.parse_known_args(argv)
    if args.command == "score":
        from .evaluation.cli import score_main
        return score_main(rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    device = _device(args.device)

    from .engine import workflow
    if args.command == "inspect":
        _setup(device)
        workflow.software_check(args.dataset, args.out, device)
    elif args.command == "train":
        config = (load_config(args.config) if args.config.is_file() else TrainConfig()).updated(**_parse_overrides(args.set))
        run = workflow.Run(args.run)
        _setup(device)
        checkpoint = workflow.develop(config, args.dataset, run, device, search=args.search)
        print("\nValidation: choosing NMS IoU and operating confidence", flush=True)
        workflow.validate(checkpoint, args.dataset, run, device, batch=config.batch)
        if args.final_test:
            print("\nFinal retrain and test", flush=True)
            workflow.final_test(run, args.dataset, device)
        print(f"\nRun folder: {run.root.resolve()}")
    elif args.command == "test":
        _setup(device)
        workflow.final_test(workflow.Run(args.run), args.dataset, device)
    elif args.command == "predict":
        from .engine.predictor import YOLOv2Detector
        from .evaluation.records import load_json
        from .visualize import draw_detections
        from PIL import Image, ImageOps

        run = workflow.Run(args.run)
        detector = YOLOv2Detector.from_checkpoint(run.best_checkpoint(), device, load_json(run.protocol_path))
        out = args.out or run.root / "predictions"
        out.mkdir(parents=True, exist_ok=True)
        for path in args.images:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            detections = detector.predict_image(image)
            draw_detections(image, detections).save(out / f"{path.stem}_detections.jpg", quality=92)
            print(f"{path.name}: " + (", ".join(f"{d['label']} {d['confidence']:.2f}" for d in detections) or "nothing detected"))
        print(f"Saved to {out.resolve()}")
    elif args.command == "gui":
        from .gui.server import serve
        serve(args.port, run=args.run, device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
