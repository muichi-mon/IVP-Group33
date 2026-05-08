from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def discover_inputs(preds_dir: Path) -> list[Path]:
    files = sorted(p for p in preds_dir.glob("*.npy")
                   if p.stem != "test_ids" and not p.stem.endswith("_ids")
                   and not p.stem.endswith("_labels"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="*", default=None)
    ap.add_argument("--preds-dir", type=Path, default=ROOT / "preds")
    ap.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    ap.add_argument("--sample", type=Path, default=ROOT / "data" / "sample_submission.csv")
    ap.add_argument("--test-csv", type=Path, default=ROOT / "data" / "test.csv")
    args = ap.parse_args()

    files = args.inputs if args.inputs else discover_inputs(args.preds_dir)

    for f in files:
        print(f"  - {f.name}")

    summed = None
    for f in files:
        arr = np.load(f)
        summed = arr if summed is None else summed + arr
    avg = summed / len(files)
    preds = avg.argmax(axis=1)

    test_ids_path = args.preds_dir / "test_ids.npy"
    test_ids = np.load(test_ids_path)

    test_csv = pd.read_csv(args.test_csv)
    pred_map = dict(zip(test_ids.tolist(), preds.tolist()))
    test_csv["Category"] = test_csv["Id"].astype(int).map(pred_map).astype(int)

    if args.sample.exists():
        sample = pd.read_csv(args.sample)
        if "Id" in sample.columns:
            sample_ids = sample["Id"].astype(int).tolist()
            current_ids = test_csv["Id"].astype(int).tolist()
            if set(sample_ids) == set(current_ids) and sample_ids != current_ids:
                test_csv = test_csv.set_index("Id").loc[sample_ids].reset_index()

    test_csv.to_csv(args.out, index=False)
    print(test_csv["Category"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
