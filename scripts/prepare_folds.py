from __future__ import annotations

import argparse
from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from birdclef2026.config import ensure_dir, load_config
from birdclef2026.data import read_metadata
from birdclef2026.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/effnet_b0.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    data_root = Path(args.data_root or cfg["data_root"])
    processed_dir = ensure_dir(cfg.get("processed_dir", "data/processed"))
    output_path = Path(args.output) if args.output else processed_dir / "train_folds.csv"

    df, _, _ = read_metadata(data_root)
    folds_cfg = cfg.get("folds", {})
    n_splits = int(folds_cfg.get("n_splits", 5))
    group_col = folds_cfg.get("group_column", "filepath")
    df["fold"] = -1

    if group_col in df.columns:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.get("seed", 42)))
        split_iter = splitter.split(df, df["primary_label"], groups=df[group_col])
    else:
        labels = sorted(df["primary_label"].unique())
        label_to_idx = {label: i for i, label in enumerate(labels)}
        y = np.zeros((len(df), len(labels)), dtype=np.int8)
        for i, row in df.iterrows():
            y[i, label_to_idx[row["primary_label"]]] = 1
            for label in row["secondary_labels"]:
                if label in label_to_idx:
                    y[i, label_to_idx[label]] = 1
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.get("seed", 42)))
        split_iter = splitter.split(df, y)

    for fold, (_, valid_idx) in enumerate(split_iter):
        df.loc[valid_idx, "fold"] = fold

    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows with {n_splits} folds to {output_path}")


if __name__ == "__main__":
    main()
