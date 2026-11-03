"""Random hyperparameter search over learning rate, weight decay and batch size (proposal section 4.2)."""

import csv
import itertools
import random
from pathlib import Path

from ..config import SEED
from .trainer import train

# %% search
SEARCH_GRID = {"lr": (1e-4, 3e-4, 1e-3), "weight_decay": (1e-4, 5e-4), "batch": (4, 8)}


def search_configurations(count, seed=SEED):
    """`count` distinct (lr, weight_decay, batch) combinations from SEARCH_GRID, in a seeded random order."""
    combinations = list(itertools.product(*SEARCH_GRID.values()))
    if not 1 <= count <= len(combinations):
        raise ValueError(f"Choose between 1 and {len(combinations)} search configurations.")
    random.Random(seed).shuffle(combinations)
    return [dict(zip(SEARCH_GRID, values)) for values in combinations[:count]]


def hyperparameter_search(config, count, dataset_root, work_dir, report_dir, device):
    """Train each sampled configuration and keep the best by validation AP50.

    Ties go to the lower validation loss at the best epoch. Every other setting is the same for all
    trials. Writes search.csv and returns the selected development checkpoint.
    """
    work_dir, report_dir = Path(work_dir), Path(report_dir)
    rows = []
    configurations = search_configurations(count)
    for number, settings in enumerate(configurations, 1):
        name = f"trial_{number:02d}"
        print(f"\nSearch {name} ({number}/{len(configurations)}): lr={settings['lr']:g} | "
              f"weight_decay={settings['weight_decay']:g} | batch={settings['batch']}", flush=True)
        summary = train(config.updated(**settings), dataset_root, work_dir / name, report_dir / name, device)
        rows.append({"trial": name, **settings, "best_map50": summary["best_map50"],
                     "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
                     "val_loss_at_best": summary["val_loss_at_best"], "checkpoint": str(summary["checkpoint"])})
    best = max(rows, key=lambda row: (row["best_map50"], -(row["val_loss_at_best"] or 0.0)))
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSearch results (validation AP50):", flush=True)
    for row in sorted(rows, key=lambda row: -row["best_map50"]):
        print(f"  {row['trial']}: lr={row['lr']:g} weight_decay={row['weight_decay']:g} batch={row['batch']} "
              f"-> AP50={row['best_map50']:.4f} (epoch {row['best_epoch']}/{row['epochs_run']})", flush=True)
    print(f"Selected {best['trial']}; table saved to {report_dir / 'search.csv'}", flush=True)
    return Path(best["checkpoint"])
