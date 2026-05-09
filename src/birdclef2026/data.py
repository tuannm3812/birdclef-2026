from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


def read_metadata(data_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    data_root = Path(data_root)
    train = pd.read_csv(data_root / "train.csv")
    taxonomy = _read_optional_csv(data_root / "taxonomy.csv")
    soundscape_labels = _read_optional_csv(data_root / "train_soundscapes_labels.csv")

    train = train.copy()
    train["filepath"] = train["filename"].map(lambda x: str(data_root / "train_audio" / x))
    train["secondary_labels"] = train.get("secondary_labels", "[]").map(parse_label_list)

    if taxonomy is not None and "primary_label" in taxonomy.columns:
        train = train.merge(taxonomy, on="primary_label", how="left", suffixes=("", "_taxonomy"))

    return train, taxonomy, soundscape_labels


def parse_label_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return []


def label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label_to_idx = {label: i for i, label in enumerate(sorted(labels))}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None
