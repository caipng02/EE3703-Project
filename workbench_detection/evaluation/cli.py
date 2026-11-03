"""Score saved prediction JSON files (for example from another detector) with the project protocol."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .metrics import EvaluationConfig
from .protocol import run_evaluation
from .records import load_ground_truth, load_json, save_json


def score_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ground-truth", type=Path, help="Ground-truth JSON in the documented common format.")
    mode.add_argument("--dataset-root", type=Path, help="Dataset folder containing manifest.csv, train/, valid/, contract/.")
    parser.add_argument("--dataset-split", choices=("train", "valid", "contract", "test"), default="valid")
    parser.add_argument("--annotation-format", choices=("coco", "yolo"), default="coco", help="Both are checked; select which precision to evaluate.")
    parser.add_argument("--allow-test", action="store_true", help="Unlock the test split once every setting is fixed.")
    parser.add_argument("--prepare-only", action="store_true", help="Load and check the split, save ground_truth.json, and stop.")
    parser.add_argument("--project-protocol", action="store_true", help="Use COCO AP and validation-selected operating confidence.")
    parser.add_argument("--select-nms", action="store_true", help="Compare validation exports for every proposed NMS candidate.")
    parser.add_argument("--frozen-protocol", type=Path, help="Validation evaluation_protocol.json required for project test.")
    parser.add_argument("--predictions", nargs="+", type=Path, help="One prediction JSON path per model.")
    parser.add_argument("--output", type=Path, help="Directory for protocol and detailed diagnostic files.")
    parser.add_argument("--results", type=Path, help="Directory for report metrics and figures.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation", help="Record intended split; does not verify provenance.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Operating threshold for P/R/F1, errors and gallery, not AP.")
    parser.add_argument("--iou", type=float, default=0.5, help="Operating IoU for P/R/F1 and errors; AP grid stays 0.50:0.05:0.95.")
    parser.add_argument("--ap-score-floor", type=float, default=None, help="Minimum supplied score used for AP; record any upstream filtering too.")
    parser.add_argument("--image-root", type=Path, help="Optional folder containing original image files for overlays.")
    parser.add_argument("--detailed", action="store_true", help="Also write CSVs, HTML, snapshots and the full gallery.")
    parser.add_argument("--no-subsets", action="store_true", help="Skip source/tag subset evaluation.")
    args = parser.parse_args(argv)
    if (args.prepare_only or args.allow_test or args.dataset_split != "valid" or args.annotation_format != "coco") and not args.dataset_root:
        parser.error("Dataset options require --dataset-root.")
    if args.prepare_only and args.predictions:
        parser.error("--prepare-only cannot consume model predictions.")
    if (args.select_nms or args.frozen_protocol) and not args.project_protocol:
        parser.error("--select-nms and --frozen-protocol require --project-protocol.")
    if args.project_protocol and args.prepare_only:
        parser.error("--project-protocol requires real model predictions.")
    if args.project_protocol and (args.confidence != 0.5 or args.iou != 0.5):
        parser.error("Project operating confidence comes from validation and matching IoU is fixed at 0.50.")
    if args.frozen_protocol and args.ap_score_floor is not None:
        parser.error("Final test reads the AP floor from the frozen protocol.")
    try:
        base = Path(__file__).resolve().parent
        floor = args.ap_score_floor if args.ap_score_floor is not None else (0.001 if args.project_protocol else 0.0)
        config = EvaluationConfig(operating_iou=args.iou, operating_confidence=args.confidence, ap_score_floor=floor)
        if args.dataset_root:
            gt = load_ground_truth(args.dataset_root, args.dataset_split,
                                           annotation_format=args.annotation_format, allow_test=args.allow_test)
            declared_split = args.dataset_split
            image_root = args.image_root or args.dataset_root / args.dataset_split / "images"
        else:
            gt = load_json(args.ground_truth)
            declared_split, image_root = args.split, args.image_root
        if args.prepare_only:
            output = args.output or base / "prepared" / args.dataset_split
            save_json(output / "ground_truth.json", gt)
            print(f"Prepared {len(gt['images'])} real images; no detector evaluated.")
            print(f"Ground truth: {(output / 'ground_truth.json').resolve()}")
            return 0
        if not args.predictions:
            parser.error("Evaluation requires --predictions; use --prepare-only to load the ground truth alone.")
        preds = [load_json(p) for p in args.predictions]
        output = args.output or base / "outputs" / "evaluation" / declared_split
        report = args.results or base / "results" / "evaluation" / declared_split
        if args.dataset_root and (args.no_subsets or args.split != "validation"):
            parser.error("Dataset evaluation reports origin subsets; select the split with --dataset-split.")
        run_evaluation(gt, preds, output, config, split=declared_split,
                                 image_root=image_root, include_subsets=not args.no_subsets,
                                 project_protocol=args.project_protocol, select_nms=args.select_nms, detailed=args.detailed, results_dir=report,
                                 frozen_protocol=load_json(args.frozen_protocol) if args.frozen_protocol else None)
        if args.detailed:
            print(f'Detailed diagnostics: {(output / "index.html").resolve()}')
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
