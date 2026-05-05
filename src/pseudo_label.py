"""Generate pseudo-labels from a per-test-id softmax matrix.

Reads either an explicit averaged npy (preferred) or computes the average
across all preds/*.npy. Writes data/pseudo.csv with columns Id,Category,confidence.

Usage:
    python -m src.pseudo_label --threshold 0.99
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", type=Path, default=ROOT / "preds")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pseudo.csv")
    args = ap.parse_args()

    files = sorted(p for p in args.preds_dir.glob("*.npy")
                   if p.stem != "test_ids" and not p.stem.endswith("_ids")
                   and not p.stem.endswith("_labels"))
    assert files, f"no prediction npy files found in {args.preds_dir}"
    summed = None
    for f in files:
        arr = np.load(f)
        summed = arr if summed is None else summed + arr
    avg = summed / len(files)

    test_ids = np.load(args.preds_dir / "test_ids.npy")
    max_prob = avg.max(axis=1)
    preds = avg.argmax(axis=1)

    keep = max_prob >= args.threshold
    df = pd.DataFrame({
        "Id": test_ids[keep].astype(int),
        "Category": preds[keep].astype(int),
        "confidence": max_prob[keep],
    })
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(df)}/{len(test_ids)} test ids kept "
          f"(threshold={args.threshold})")
    if len(df) > 0:
        print("Pseudo-label class distribution:")
        print(df["Category"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
