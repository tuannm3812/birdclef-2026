from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from birdclef2026.config import ensure_dir, load_config
from birdclef2026.data import read_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/effnet_b0.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg["data_root"])
    output_dir = ensure_dir(args.output_dir or Path(cfg.get("output_dir", "outputs")) / "eda")

    df, taxonomy, soundscape_labels = read_metadata(data_root)

    label_counts = (
        df["primary_label"]
        .value_counts()
        .rename_axis("primary_label")
        .reset_index(name="recordings")
    )
    label_counts.to_csv(output_dir / "primary_label_counts.csv", index=False)

    duration_summary = df["duration"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]) if "duration" in df else pd.Series(dtype=float)
    duration_summary.to_csv(output_dir / "duration_summary.csv", header=["value"])

    print(f"Rows: {len(df):,}")
    print(f"Classes: {df['primary_label'].nunique():,}")
    print(f"Top labels:\n{label_counts.head(10).to_string(index=False)}")
    if taxonomy is not None:
        print(f"Taxonomy rows: {len(taxonomy):,}")
    if soundscape_labels is not None:
        print(f"Soundscape label rows: {len(soundscape_labels):,}")
    print(f"Wrote EDA tables to {output_dir}")


if __name__ == "__main__":
    main()
