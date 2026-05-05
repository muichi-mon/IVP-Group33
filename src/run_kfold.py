"""Run all 5 folds for one architecture and report CV accuracy.

Usage:
    python -m src.run_kfold --config configs/convnext_tiny.yaml
    python -m src.run_kfold --config configs/convnext_tiny.yaml --pseudo data/pseudo.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.train import train_one_fold

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--pseudo", type=Path, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    arch = cfg["arch"]

    results = []
    for fold in args.folds:
        r = train_one_fold(cfg, fold, debug=args.debug, pseudo_csv=args.pseudo)
        results.append(r)

    df = pd.DataFrame(results)
    print(f"\n=== {arch} CV summary ===")
    print(df.to_string(index=False))
    print(f"Mean val_acc: {df['best_val_acc'].mean():.4f}  "
          f"(std {df['best_val_acc'].std():.4f})")

    # Aggregate OOF and report a single CV accuracy
    oof_dir = ROOT / "oof"
    all_probs, all_ids, all_labels = [], [], []
    for fold in args.folds:
        all_probs.append(np.load(oof_dir / f"{arch}_fold{fold}.npy"))
        all_ids.append(np.load(oof_dir / f"{arch}_fold{fold}_ids.npy"))
        all_labels.append(np.load(oof_dir / f"{arch}_fold{fold}_labels.npy"))
    probs = np.concatenate(all_probs, axis=0)
    ids = np.concatenate(all_ids, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    cv_acc = (probs.argmax(axis=1) == labels).mean()
    print(f"OOF accuracy across all folds: {cv_acc:.4f}")

    np.save(oof_dir / f"{arch}_oof.npy", probs)
    np.save(oof_dir / f"{arch}_oof_ids.npy", ids)
    np.save(oof_dir / f"{arch}_oof_labels.npy", labels)


if __name__ == "__main__":
    main()
