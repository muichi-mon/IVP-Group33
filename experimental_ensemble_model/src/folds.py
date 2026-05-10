"""Stratified 5-fold split on original training IDs.

Reads data/train.csv (Id, Category) and writes data/folds.csv (Id, Category, fold).
Splits operate on original IDs only — augmented copies (if any) inherit their
original's fold via Id matching at training time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def make_folds(n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(DATA / "train.csv")
    assert {"Id", "Category"}.issubset(df.columns), df.columns
    df = df.reset_index(drop=True)
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df["Id"], df["Category"])):
        df.loc[val_idx, "fold"] = fold

    assert (df["fold"] >= 0).all(), "some rows didn't get assigned a fold"
    return df


def verify(df: pd.DataFrame, n_splits: int) -> None:
    counts = df.groupby(["fold", "Category"]).size().unstack(fill_value=0)
    print("Fold × Category counts:")
    print(counts)
    fold_sizes = df["fold"].value_counts().sort_index()
    print(f"\nFold sizes: {fold_sizes.to_dict()}")
    assert df["fold"].nunique() == n_splits
    assert all(counts.values.flatten() > 0), "every (fold, class) cell must be populated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DATA / "folds.csv")
    args = ap.parse_args()

    df = make_folds(args.n_splits, args.seed)
    verify(df, args.n_splits)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
