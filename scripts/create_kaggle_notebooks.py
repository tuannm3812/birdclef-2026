from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path("notebooks")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


def lines(source: str) -> list[str]:
    source = source.strip("\n")
    return [line + "\n" for line in source.splitlines()]


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_STYLE = """
from pathlib import Path
import json
import os
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)


class CFG:
    seed = 42
    competition_name = "birdclef-2026"
    data_root = None
    artifact_dir = Path("/kaggle/working/artifacts")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def find_data_root() -> Path:
    candidates = [
        Path("/kaggle/input/birdclef-2026"),
        Path("/kaggle/input/birdclef-2026-repack/birdclef-2026"),
        Path("/kaggle/input/birdclef-2026-repack"),
        Path("data/raw/birdclef-2026"),
    ]
    for path in candidates:
        if (path / "train.csv").exists():
            return path
    input_root = Path("/kaggle/input")
    if input_root.exists():
        matches = list(input_root.glob("**/train.csv"))
        if matches:
            return matches[0].parent
    raise FileNotFoundError("Could not find train.csv. Attach the BirdCLEF 2026 dataset.")


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


seed_everything(CFG.seed)
CFG.data_root = find_data_root()
CFG.artifact_dir.mkdir(parents=True, exist_ok=True)

print(f"Data root: {CFG.data_root}")
print(f"Artifacts: {CFG.artifact_dir}")
"""


def eda_notebook() -> dict:
    return notebook(
        [
            md(
                """
# BirdCLEF+ 2026 - Insightful EDA

**Purpose.** Build a clear picture of the BirdCLEF+ 2026 training data before modeling: label imbalance, taxonomy coverage, duration/chunking behavior, secondary-label noise, soundscape annotations, and representative audio examples.

**Run mode.** Kaggle analysis notebook; no model training.

**Primary outputs.** CSV summaries, diagnostic plots, representative mel-spectrograms, and a zipped artifact bundle under `/kaggle/working`.

Artifacts are written to `/kaggle/working/artifacts/eda`.
"""
            ),
            md("## 1. Setup"),
            code(
                COMMON_STYLE
                + """
import ast
from collections import Counter

import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio, display

CFG.artifact_dir = CFG.artifact_dir / "eda"
CFG.artifact_dir.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


class CFG(CFG):
    sample_rate = 32000
    clip_seconds = 5
    n_mels = 128
    top_n = 30
    random_examples = 6
"""
            ),
            md("## 2. Load Metadata"),
            md(
                """
**Insight goal.** Before modeling, confirm the competition files are mounted correctly and that the training table, taxonomy table, soundscape labels, and sample submission agree with each other. This section creates the basic dataset inventory that later sections use for imbalance, duration, and domain-shift diagnostics.
"""
            ),
            code(
                """
def parse_label_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


train = pd.read_csv(CFG.data_root / "train.csv")
taxonomy = read_optional_csv(CFG.data_root / "taxonomy.csv")
soundscape_labels = read_optional_csv(CFG.data_root / "train_soundscapes_labels.csv")
sample_submission = read_optional_csv(CFG.data_root / "sample_submission.csv")

train["filepath"] = train["filename"].map(lambda x: CFG.data_root / "train_audio" / x)
if "secondary_labels" in train.columns:
    train["secondary_labels"] = train["secondary_labels"].map(parse_label_list)
else:
    train["secondary_labels"] = [[] for _ in range(len(train))]

if taxonomy is not None and "primary_label" in taxonomy.columns:
    train = train.merge(taxonomy, on="primary_label", how="left", suffixes=("", "_taxonomy"))

summary = {
    "train_rows": len(train),
    "primary_classes": train["primary_label"].nunique(),
    "taxonomy_rows": 0 if taxonomy is None else len(taxonomy),
    "soundscape_label_rows": 0 if soundscape_labels is None else len(soundscape_labels),
    "sample_submission_rows": 0 if sample_submission is None else len(sample_submission),
    "audio_files_found": int(train["filepath"].map(Path.exists).sum()),
}
pd.Series(summary).to_csv(CFG.artifact_dir / "dataset_summary.csv", header=["value"])
display(pd.Series(summary).to_frame("value"))
display(train.head())
"""
            ),
            md("## 3. Dataset Schema And Missingness"),
            md(
                """
**What to look for.** Missing metadata can create silent leakage or brittle preprocessing logic. The most important fields for the baseline notebooks are `filename`, `primary_label`, `secondary_labels`, and `duration`; if any of those are incomplete, the training pipeline needs explicit fallback behavior.
"""
            ),
            code(
                """
def safe_nunique(series: pd.Series) -> int:
    values = series.map(lambda x: tuple(x) if isinstance(x, list) else x)
    return int(values.nunique(dropna=True))


schema = pd.DataFrame(
    {
        "column": train.columns,
        "dtype": [str(train[col].dtype) for col in train.columns],
        "missing": [int(train[col].isna().sum()) for col in train.columns],
        "missing_pct": [float(train[col].isna().mean()) for col in train.columns],
        "unique": [safe_nunique(train[col]) for col in train.columns],
    }
)
schema.to_csv(CFG.artifact_dir / "train_schema_missingness.csv", index=False)
display(schema)

missing_files = train.loc[~train["filepath"].map(Path.exists), ["filename", "filepath"]]
missing_files.to_csv(CFG.artifact_dir / "missing_audio_files.csv", index=False)
print(f"Missing audio files: {len(missing_files):,}")
"""
            ),
            md("## 4. Duplicate Records"),
            md(
                """
**Why this matters.** Duplicate metadata rows can inflate apparent sample size and leak repeated labels into validation. The soundscape label file is especially important because exact duplicate 5-second annotations can bias multi-label prevalence estimates.
"""
            ),
            code(
                """
duplicate_tables = {}
for name, frame in {
    "train": train,
    "taxonomy": taxonomy,
    "soundscape_labels": soundscape_labels,
    "sample_submission": sample_submission,
}.items():
    if frame is None:
        continue
    dup_mask = frame.duplicated(keep=False)
    duplicate_tables[name] = int(dup_mask.sum())
    frame.loc[dup_mask].to_csv(CFG.artifact_dir / f"{name}_duplicate_rows.csv", index=False)

duplicate_summary = pd.Series(duplicate_tables, name="duplicate_rows").rename_axis("table").reset_index()
duplicate_summary.to_csv(CFG.artifact_dir / "duplicate_summary.csv", index=False)
display(duplicate_summary)

key_checks = []
for name, frame, candidates in [
    ("train", train, [["filename"], ["filepath"], ["primary_label", "filename"]]),
    ("taxonomy", taxonomy, [["primary_label"], ["scientific_name"]]),
    ("soundscape_labels", soundscape_labels, [["row_id"], ["filename"], ["filename", "primary_label"]]),
    ("sample_submission", sample_submission, [["row_id"]]),
]:
    if frame is None:
        continue
    for keys in candidates:
        if all(key in frame.columns for key in keys):
            dup_count = int(frame.duplicated(subset=keys, keep=False).sum())
            key_checks.append({"table": name, "keys": "+".join(keys), "duplicate_rows": dup_count})
            if dup_count:
                frame.loc[frame.duplicated(subset=keys, keep=False)].sort_values(keys).to_csv(
                    CFG.artifact_dir / f"{name}_{'_'.join(keys)}_duplicates.csv",
                    index=False,
                )

key_duplicate_summary = pd.DataFrame(key_checks)
key_duplicate_summary.to_csv(CFG.artifact_dir / "key_duplicate_summary.csv", index=False)
display(key_duplicate_summary)

if soundscape_labels is not None:
    soundscape_dedup = soundscape_labels.drop_duplicates().reset_index(drop=True)
    print(f"Soundscape labels before deduplication: {len(soundscape_labels):,}")
    print(f"Soundscape labels after deduplication: {len(soundscape_dedup):,}")
else:
    soundscape_dedup = None
"""
            ),
            md(
                """
**Takeaway.** If duplicate rows appear in `train_soundscapes_labels.csv`, downstream soundscape prevalence and co-occurrence analysis should use the deduplicated table. The original file is still copied to artifacts for auditability.
"""
            ),
            md("## 5. Primary Label Imbalance"),
            md(
                """
**Why this matters.** BirdCLEF-style datasets are usually long-tailed: common species dominate the loss, while rare species are easy to ignore. The imbalance plots below help decide whether the first baseline should use balanced sampling, class-aware augmentation, focal loss, or per-class validation diagnostics.
"""
            ),
            code(
                """
label_counts = (
    train["primary_label"]
    .value_counts()
    .rename_axis("primary_label")
    .reset_index(name="recordings")
)
label_counts["share"] = label_counts["recordings"] / label_counts["recordings"].sum()
label_counts["cumulative_share"] = label_counts["share"].cumsum()
label_counts["imbalance_ratio_vs_median"] = label_counts["recordings"] / label_counts["recordings"].median()
label_counts.to_csv(CFG.artifact_dir / "primary_label_counts.csv", index=False)

head_share = label_counts.head(10)["share"].sum()
tail_singletons = int((label_counts["recordings"] == 1).sum())
print(f"Top 10 classes contain {head_share:.1%} of recordings.")
print(f"Singleton classes: {tail_singletons:,}")
display(label_counts.head(CFG.top_n))
display(label_counts.tail(CFG.top_n))

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=label_counts.head(CFG.top_n), x="recordings", y="primary_label", ax=ax, color="#3C78D8")
ax.set_title(f"Top {CFG.top_n} primary labels by recording count")
ax.set_xlabel("recordings")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(CFG.artifact_dir / "top_primary_labels.png", dpi=160)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(label_counts["recordings"], bins=50, ax=axes[0], color="#2CA02C")
axes[0].set_title("Class-count distribution")
axes[0].set_xlabel("recordings per class")
sns.lineplot(data=label_counts.reset_index(), x="index", y="cumulative_share", ax=axes[1], color="#D62728")
axes[1].set_title("Cumulative share by ranked class")
axes[1].set_xlabel("class rank")
axes[1].set_ylabel("cumulative share")
fig.tight_layout()
fig.savefig(CFG.artifact_dir / "class_imbalance_diagnostics.png", dpi=160)
plt.show()
"""
            ),
            md("## 6. Duration And Chunking Implications"),
            md(
                """
**Why this matters.** A 5-second crop is a modeling choice, not just a preprocessing detail. Short clips may need padding, long recordings can provide many training crops, and classes with fewer files but longer total duration may be less data-poor than raw recording counts suggest.
"""
            ),
            code(
                """
if "duration" in train.columns:
    duration_summary = train["duration"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    duration_summary.to_csv(CFG.artifact_dir / "duration_summary.csv", header=["value"])
    display(duration_summary)

    duration_by_label = (
        train.groupby("primary_label")["duration"]
        .agg(recordings="count", median_duration="median", total_seconds="sum")
        .reset_index()
        .sort_values("total_seconds", ascending=False)
    )
    duration_by_label["estimated_5s_chunks"] = np.ceil(duration_by_label["total_seconds"] / CFG.clip_seconds).astype(int)
    duration_by_label.to_csv(CFG.artifact_dir / "duration_by_primary_label.csv", index=False)
    display(duration_by_label.head(20))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    clipped = train["duration"].clip(upper=train["duration"].quantile(0.99))
    sns.histplot(clipped, bins=60, ax=axes[0], color="#9467BD")
    axes[0].set_title("Audio duration distribution, clipped at p99")
    axes[0].set_xlabel("seconds")
    sns.scatterplot(
        data=duration_by_label,
        x="recordings",
        y="total_seconds",
        size="median_duration",
        sizes=(20, 180),
        alpha=0.7,
        ax=axes[1],
        color="#FF7F0E",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Per-class recording count vs total audio")
    axes[1].set_xlabel("recordings, log scale")
    axes[1].set_ylabel("total seconds, log scale")
    fig.tight_layout()
    fig.savefig(CFG.artifact_dir / "duration_and_chunking.png", dpi=160)
    plt.show()
else:
    print("No duration column found in train.csv.")
"""
            ),
            md("## 7. Secondary Labels And Co-Occurrence"),
            md(
                """
**Why this matters.** Secondary labels are noisy but valuable. They reveal species that often appear together and can later support soft labels, multi-label training, mixup targets, or post-processing rules. For the first EfficientNet baseline, they are kept diagnostic rather than target-defining.
"""
            ),
            code(
                """
secondary = train[["filename", "primary_label", "secondary_labels"]].explode("secondary_labels")
secondary = secondary.dropna(subset=["secondary_labels"])
secondary_counts = (
    secondary["secondary_labels"]
    .value_counts()
    .rename_axis("secondary_label")
    .reset_index(name="mentions")
)
secondary_counts.to_csv(CFG.artifact_dir / "secondary_label_counts.csv", index=False)
rows_with_secondary = int((train["secondary_labels"].map(len) > 0).sum())
print(f"Rows with secondary labels: {rows_with_secondary:,} ({rows_with_secondary / len(train):.1%})")
display(secondary_counts.head(CFG.top_n))

cooccurrence = (
    secondary.groupby(["primary_label", "secondary_labels"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
cooccurrence.to_csv(CFG.artifact_dir / "primary_secondary_cooccurrence.csv", index=False)
display(cooccurrence.head(30))

if len(secondary_counts):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=secondary_counts.head(CFG.top_n), x="mentions", y="secondary_label", ax=ax, color="#17BECF")
    ax.set_title(f"Top {CFG.top_n} secondary labels")
    ax.set_xlabel("mentions")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(CFG.artifact_dir / "top_secondary_labels.png", dpi=160)
    plt.show()
"""
            ),
            md("## 8. Taxonomy Coverage"),
            md(
                """
**Why this matters.** Taxonomy metadata gives a way to audit errors above the species level. If the model confuses species within the same genus or family, that is a different failure mode from confusing unrelated calls. This table also checks whether train labels and taxonomy labels are aligned.
"""
            ),
            code(
                """
if taxonomy is not None:
    taxonomy.to_csv(CFG.artifact_dir / "taxonomy_copy.csv", index=False)
    display(taxonomy.head())
    taxonomy_cols = taxonomy.columns.tolist()
    print(taxonomy_cols)

    taxonomy_label_col = "primary_label" if "primary_label" in taxonomy.columns else None
    if taxonomy_label_col:
        train_labels = set(pd.read_csv(CFG.data_root / "train.csv")["primary_label"].unique())
        taxonomy_labels = set(taxonomy[taxonomy_label_col].dropna().astype(str).unique())
        coverage = {
            "train_labels": len(train_labels),
            "taxonomy_labels": len(taxonomy_labels),
            "train_labels_missing_from_taxonomy": len(train_labels - taxonomy_labels),
            "taxonomy_labels_not_in_train": len(taxonomy_labels - train_labels),
        }
        pd.Series(coverage).to_csv(CFG.artifact_dir / "taxonomy_coverage.csv", header=["value"])
        display(pd.Series(coverage).to_frame("value"))

    candidate_cols = [c for c in ["class_name", "order", "family", "genus", "species", "common_name", "scientific_name"] if c in train.columns]
    for col in candidate_cols[:4]:
        counts = train[col].value_counts(dropna=False).head(20).rename_axis(col).reset_index(name="recordings")
        display(counts)
else:
    print("No taxonomy.csv found.")
"""
            ),
            md("## 9. Metadata Quality And Geography"),
            md(
                """
**Why this matters.** Metadata quality is uneven across recording sources. Rating, collection, author, license, and geography can all become hidden confounders: a model may learn recorder/source artifacts instead of species-specific acoustic structure.
"""
            ),
            code(
                """
metadata_outputs = {}

if "rating" in train.columns:
    rating_counts = train["rating"].value_counts(dropna=False).sort_index().rename_axis("rating").reset_index(name="recordings")
    rating_counts.to_csv(CFG.artifact_dir / "rating_counts.csv", index=False)
    display(rating_counts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=rating_counts, x="rating", y="recordings", ax=axes[0], color="#7F7F7F")
    axes[0].set_title("Recording rating distribution")
    if "class_name" in train.columns:
        quality_by_class = train.groupby("class_name")["rating"].agg(["count", "mean", "median"]).reset_index()
        quality_by_class.to_csv(CFG.artifact_dir / "quality_by_class.csv", index=False)
        sns.boxplot(data=train, x="rating", y="class_name", ax=axes[1], color="#BCBD22")
        axes[1].set_title("Rating by biological class")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(CFG.artifact_dir / "metadata_quality_rating.png", dpi=160)
    plt.show()

for col in ["collection", "license", "type", "author"]:
    if col in train.columns:
        counts = train[col].value_counts(dropna=False).head(30).rename_axis(col).reset_index(name="recordings")
        counts.to_csv(CFG.artifact_dir / f"{col}_counts.csv", index=False)
        metadata_outputs[col] = len(counts)

if {"latitude", "longitude"}.issubset(train.columns):
    pantanal = {"lat_min": -21.6, "lat_max": -16.5, "lon_min": -57.6, "lon_max": -55.9}
    geo = train.dropna(subset=["latitude", "longitude"]).copy()
    geo["inside_pantanal_box"] = (
        geo["latitude"].between(pantanal["lat_min"], pantanal["lat_max"])
        & geo["longitude"].between(pantanal["lon_min"], pantanal["lon_max"])
    )
    geo_summary = pd.Series(
        {
            "records_with_coordinates": len(geo),
            "inside_pantanal_box": int(geo["inside_pantanal_box"].sum()),
            "inside_pantanal_share": float(geo["inside_pantanal_box"].mean()),
            "species_inside_pantanal": int(geo.loc[geo["inside_pantanal_box"], "primary_label"].nunique()),
        }
    )
    geo_summary.to_csv(CFG.artifact_dir / "geography_summary.csv", header=["value"])
    display(geo_summary.to_frame("value"))

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=geo.sample(min(len(geo), 12000), random_state=CFG.seed),
        x="longitude",
        y="latitude",
        hue="inside_pantanal_box",
        s=8,
        alpha=0.35,
        ax=ax,
    )
    ax.set_title("Training recording geography")
    fig.tight_layout()
    fig.savefig(CFG.artifact_dir / "recording_geography.png", dpi=160)
    plt.show()
"""
            ),
            md(
                """
**Takeaway.** Treat quality/source/geography as potential domain-shift variables. Validation splits grouped by author, recording, or source can be more honest than purely random splits, especially when the hidden soundscapes come from a narrower ecological region.
"""
            ),
            md("## 10. Soundscape Labels"),
            md(
                """
**Why this matters.** Soundscapes are closer to the evaluation domain than clean training clips: longer recordings, overlapping calls, background noise, and sparse temporal annotations. Treating this table as a separate domain helps avoid over-trusting validation metrics from clean clips only.
"""
            ),
            code(
                """
if soundscape_labels is None:
    print("No train_soundscapes_labels.csv found.")
else:
    soundscape_labels.to_csv(CFG.artifact_dir / "soundscape_labels_copy.csv", index=False)
    soundscape_analysis = soundscape_dedup if "soundscape_dedup" in globals() and soundscape_dedup is not None else soundscape_labels
    display(soundscape_analysis.head())
    print(soundscape_analysis.columns.tolist())

    label_like_cols = [c for c in soundscape_analysis.columns if "label" in c.lower() or "species" in c.lower() or "code" in c.lower()]
    time_like_cols = [c for c in soundscape_analysis.columns if "time" in c.lower() or "second" in c.lower()]
    file_like_cols = [c for c in soundscape_analysis.columns if "filename" in c.lower() or "soundscape" in c.lower() or "row_id" in c.lower()]

    soundscape_summary = {
        "rows": len(soundscape_analysis),
        "columns": len(soundscape_analysis.columns),
        "label_like_columns": ", ".join(label_like_cols),
        "time_like_columns": ", ".join(time_like_cols),
        "file_like_columns": ", ".join(file_like_cols),
    }
    pd.Series(soundscape_summary).to_csv(CFG.artifact_dir / "soundscape_summary.csv", header=["value"])
    display(pd.Series(soundscape_summary).to_frame("value"))

    if label_like_cols:
        col = label_like_cols[0]
        sc_counts = soundscape_analysis[col].value_counts().head(CFG.top_n).rename_axis(col).reset_index(name="rows")
        sc_counts.to_csv(CFG.artifact_dir / "soundscape_label_counts.csv", index=False)
        display(sc_counts)

    if {"filename", "primary_label"}.issubset(soundscape_analysis.columns):
        sc = soundscape_analysis.copy()
        sc["label_list"] = sc["primary_label"].astype(str).str.split(";")
        sc["n_labels"] = sc["label_list"].map(len)
        sc["site"] = sc["filename"].astype(str).str.extract(r"_(S\\d+)_", expand=False)
        sc["hour"] = sc["filename"].astype(str).str.extract(r"_(\\d{6})\\.ogg$", expand=False).str[:2]
        overlap_summary = sc["n_labels"].describe(percentiles=[0.5, 0.75, 0.9]).to_frame("value")
        overlap_summary.to_csv(CFG.artifact_dir / "soundscape_overlap_summary.csv")
        display(overlap_summary)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(sc["n_labels"], discrete=True, ax=axes[0], color="#8C564B")
        axes[0].set_title("Labels per soundscape segment")
        if sc["hour"].notna().any():
            hour_counts = sc["hour"].value_counts().sort_index().rename_axis("hour").reset_index(name="segments")
            hour_counts.to_csv(CFG.artifact_dir / "soundscape_hour_counts.csv", index=False)
            sns.barplot(data=hour_counts, x="hour", y="segments", ax=axes[1], color="#E377C2")
            axes[1].set_title("Labeled soundscape segments by hour")
        else:
            axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(CFG.artifact_dir / "soundscape_overlap_time.png", dpi=160)
        plt.show()

"""
            ),
            md(
                """
**Takeaway.** Soundscape labels should be interpreted after deduplication and with time/site context. Multi-label density and temporal clustering are clues for hour-aware priors, threshold tuning, and validation design.
"""
            ),
            md("## 11. Representative Audio And Spectrograms"),
            md(
                """
**Why this matters.** A few spectrograms often reveal issues that tables hide: silence, clipping, background insects, rain, distant calls, and frequency bands that matter for augmentation. These examples are not a validation set; they are a quick sanity check for the acoustic texture the model will see.
"""
            ),
            code(
                """
def load_clip(path: Path, seconds: float = 5.0) -> np.ndarray:
    y, _ = librosa.load(path, sr=CFG.sample_rate, mono=True, duration=seconds)
    target_len = int(CFG.sample_rate * seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len].astype(np.float32)


available = train[train["filepath"].map(Path.exists)].copy()
if len(available) == 0:
    print("No local audio files available for previews.")
else:
    example_df = (
        available.groupby("primary_label", group_keys=False)
        .apply(lambda x: x.sample(1, random_state=CFG.seed))
        .sample(min(CFG.random_examples, available["primary_label"].nunique()), random_state=CFG.seed)
        .reset_index(drop=True)
    )
    example_df[["filename", "primary_label"]].to_csv(CFG.artifact_dir / "audio_examples.csv", index=False)
    display(example_df[["filename", "primary_label"]])

    fig, axes = plt.subplots(len(example_df), 1, figsize=(12, 2.6 * len(example_df)))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, example_df.iterrows()):
        y = load_clip(row["filepath"], CFG.clip_seconds)
        mel = librosa.feature.melspectrogram(y=y, sr=CFG.sample_rate, n_mels=CFG.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=CFG.sample_rate, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(f"{row['primary_label']} | {row['filename']}")
    fig.tight_layout()
    fig.savefig(CFG.artifact_dir / "representative_mels.png", dpi=160)
    plt.show()

    first = example_df.iloc[0]
    print(f"Audio preview: {first['primary_label']} | {first['filename']}")
    display(Audio(load_clip(first["filepath"], CFG.clip_seconds), rate=CFG.sample_rate))
"""
            ),
            md("## 12. Analysis Takeaways"),
            md(
                """
The diagnostics above point to a few concrete modeling decisions:

1. **Class imbalance is the first-order risk.** The ranked-class curve rises quickly, and our saved EDA artifacts show that the top 30 labels account for roughly 40% of recordings while dozens of classes have very little data. A plain random sampler will over-optimize for head classes unless we add class-aware sampling, rare-class augmentation, or per-class monitoring.

2. **Recording count is not the same as acoustic coverage.** Duration diagnostics help identify classes that have fewer files but more total seconds, and classes that have many short clips. A fixed 5-second crop is a reasonable baseline, but long clips should eventually support random crops during training and multi-crop averaging during inference.

3. **Secondary labels are useful signal, but not a clean first target.** The secondary-label plot shows repeated co-occurrence patterns. These labels are noisy enough that the first baseline should stay single-label, but they are valuable for later soft targets, mixup labels, and confusion analysis.

4. **Soundscapes are a domain shift, not just extra rows.** The soundscape label table contains multi-label combinations that look much closer to the evaluation setting than clean training clips. Validation based only on isolated `train_audio` clips can overstate leaderboard readiness.

5. **Spectrograms support both CNN and foundation-model strategies.** Representative mel plots show repeated phrases, sparse calls, background noise, and frequency-specific structure. That supports an EfficientNet mel baseline, while the stronger Perch probe result suggests Perch is best used as a teacher or feature source for a faster student model.
"""
            ),
            md(
                """
**How to use this section.** The notebook turns the diagnostics above into practical modeling implications. These takeaways should guide the baseline notebooks: validation design, crop length, augmentation strength, class balancing, and whether to compare clean-clip validation against soundscape-style artifacts.
"""
            ),
            code(
                """
takeaways = []
takeaways.append(
    f"Primary-label imbalance is substantial: top 10 classes cover {label_counts.head(10)['share'].sum():.1%} of training recordings."
)
if "duration" in train.columns:
    takeaways.append(
        f"Median clip duration is {train['duration'].median():.1f}s; a {CFG.clip_seconds}s training window creates multiple possible crops for long recordings."
    )
takeaways.append(
    f"Secondary labels appear in {(train['secondary_labels'].map(len) > 0).mean():.1%} of rows, so noisy multi-label context may help later even if the first baseline is single-label."
)
if taxonomy is not None:
    takeaways.append("Taxonomy metadata can support stratified diagnostics and class-family level error analysis.")
if soundscape_labels is not None:
    takeaways.append("Soundscape annotations should be treated separately from clean training clips because they reflect the evaluation domain more closely.")

takeaways_df = pd.DataFrame({"takeaway": takeaways})
takeaways_df.to_csv(CFG.artifact_dir / "modeling_takeaways.csv", index=False)
display(takeaways_df)
"""
            ),
            md("## 13. Final Conclusion"),
            md(
                """
This EDA suggests a practical modeling roadmap:

- Use the EfficientNet-B0 mel baseline as the dependable submission path because it is fast, pure PyTorch, and already competition-safe.
- Use Perch v2 as an offline teacher or feature generator rather than direct hidden-test inference, because the CUDA SavedModel is too slow and the CPU/cache path depends on external coverage.
- Improve validation before chasing architecture complexity: group leakage, source bias, geography drift, and soundscape overlap can all make local metrics misleading.
- Prioritize class-aware training, multi-crop inference, and eventually soundscape-informed calibration. These changes are more aligned with the observed data issues than simply scaling the backbone.
"""
            ),
            md("## 14. Artifact Manifest"),
            code(
                """
manifest = sorted(str(path.relative_to(CFG.artifact_dir)) for path in CFG.artifact_dir.glob("*"))
(CFG.artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
manifest
"""
            ),
            md("## 15. Package Artifacts For Download"),
            md(
                """
Kaggle makes files under `/kaggle/working` downloadable after the notebook finishes. This cell zips the EDA tables and figures into one file so you can download them, review them locally, and optionally commit selected lightweight artifacts such as `.csv`, `.json`, or `.png` files to GitHub.

Avoid committing large generated arrays or model checkpoints unless you intentionally manage them with Git LFS or a release asset.
"""
            ),
            code(
                """
import zipfile
from IPython.display import FileLink

zip_path = Path("/kaggle/working/birdclef_eda_artifacts.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(CFG.artifact_dir.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=path.relative_to(CFG.artifact_dir.parent))

print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")
display(FileLink(zip_path))
"""
            ),
        ]
    )


def effnet_notebook() -> dict:
    return notebook(
        [
            md(
                """
# BirdCLEF+ 2026 - Baseline EfficientNet-B0

**Purpose.** Train a fast, reproducible PyTorch baseline using 5-second mel-spectrogram crops and an EfficientNet-B0 classifier.

**Run mode.** Kaggle training notebook. Designed to be stable on Kaggle by using single-process audio loading.

**Primary outputs.** Best checkpoint, label mapping, fold assignments, training history, and a zipped artifact bundle.

Artifacts are written to `/kaggle/working/artifacts/effnet_b0`.
"""
            ),
            md("## 1. Setup"),
            code(
                COMMON_STYLE
                + """
try:
    import timm
except ImportError:
    import sys
    !{sys.executable} -m pip install -q timm
    import timm

import librosa
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(min(2, os.cpu_count() or 1))
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass


class CFG(CFG):
    artifact_dir = Path("/kaggle/working/artifacts/effnet_b0")
    sample_rate = 32000
    duration = 5.0
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20
    fmax = 16000
    n_splits = 5
    fold = 0
    backbone = "efficientnet_b0"
    pretrained = True
    epochs = 5
    batch_size = 32
    # Kaggle notebooks can crash with multiprocessing DataLoader workers when
    # audio decoding happens inside __getitem__. Keep this at 0 for stability.
    num_workers = 0
    force_single_process_loader = True
    lr = 3e-4
    weight_decay = 1e-2
    label_smoothing = 0.05
    max_train_samples = None
    max_valid_samples = None


CFG.artifact_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
"""
            ),
            md("## 2. Load Metadata And Build Folds"),
            md(
                """
This section creates a reproducible validation split and persists it with the artifacts. The first baseline uses `primary_label` only, which keeps the training objective simple and gives us a dependable reference score before adding multi-label complexity.
"""
            ),
            code(
                """
train = pd.read_csv(CFG.data_root / "train.csv")
train["filepath"] = train["filename"].map(lambda x: CFG.data_root / "train_audio" / x)
labels = sorted(train["primary_label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
train["target"] = train["primary_label"].map(label_to_idx)

train["fold"] = -1
group_col = "filename"
try:
    splitter = StratifiedGroupKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    splits = splitter.split(train, train["target"], groups=train[group_col])
except ValueError:
    splitter = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    splits = splitter.split(train, train["target"])

for fold, (_, valid_idx) in enumerate(splits):
    train.loc[valid_idx, "fold"] = fold

train.to_csv(CFG.artifact_dir / "train_folds.csv", index=False)
(CFG.artifact_dir / "labels.json").write_text(json.dumps(idx_to_label, indent=2), encoding="utf-8")

train_df = train[train["fold"] != CFG.fold].reset_index(drop=True)
valid_df = train[train["fold"] == CFG.fold].reset_index(drop=True)
if CFG.max_train_samples:
    train_df = train_df.sample(CFG.max_train_samples, random_state=CFG.seed).reset_index(drop=True)
if CFG.max_valid_samples:
    valid_df = valid_df.sample(CFG.max_valid_samples, random_state=CFG.seed).reset_index(drop=True)

print(f"Train rows: {len(train_df):,}")
print(f"Valid rows: {len(valid_df):,}")
print(f"Classes: {len(labels):,}")
"""
            ),
            md("## 3. Dataset"),
            md(
                """
Audio is converted into normalized mel-spectrograms on the fly. The implementation favors reliability over maximum throughput because Kaggle worker multiprocessing can be fragile when decoding many audio files inside `Dataset.__getitem__`.
"""
            ),
            code(
                """
def load_audio(path: Path, duration: float, train_mode: bool) -> np.ndarray:
    target_len = int(CFG.sample_rate * duration)
    offset = 0.0
    y, _ = librosa.load(path, sr=CFG.sample_rate, mono=True, offset=offset, duration=duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len].astype(np.float32)


def audio_to_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=CFG.sample_rate,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        n_mels=CFG.n_mels,
        fmin=CFG.fmin,
        fmax=CFG.fmax,
        power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.astype(np.float32)


class BirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train_mode: bool):
        self.df = df.reset_index(drop=True)
        self.train_mode = train_mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = load_audio(row["filepath"], CFG.duration, self.train_mode)
        if self.train_mode and random.random() < 0.5:
            y = y * random.uniform(0.75, 1.25)
        x = torch.from_numpy(audio_to_mel(y)).unsqueeze(0)
        target = torch.tensor(row["target"], dtype=torch.long)
        return x, target


def effective_num_workers() -> int:
    if CFG.force_single_process_loader or Path("/kaggle/working").exists():
        return 0
    return int(CFG.num_workers)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    workers = effective_num_workers()
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


print(f"Effective DataLoader workers: {effective_num_workers()}")
train_loader = make_loader(BirdDataset(train_df, train_mode=True), CFG.batch_size, True)
valid_loader = make_loader(BirdDataset(valid_df, train_mode=False), CFG.batch_size * 2, False)
"""
            ),
            md("## 4. Model And Training Loop"),
            md(
                """
EfficientNet-B0 is intentionally modest: it trains quickly, exports cleanly, and is fast enough for competition reruns. This is the submission-safe baseline we compare stronger but slower methods against.
"""
            ),
            code(
                """
class BirdClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = timm.create_model(
            CFG.backbone,
            pretrained=CFG.pretrained,
            in_chans=1,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


model = BirdClassifier(num_classes=len(labels)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
"""
            ),
            code(
                """
def train_one_epoch() -> float:
    model.train()
    total_loss = 0.0
    for x, y in tqdm(train_loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
    return total_loss / max(len(train_loader.dataset), 1)


@torch.no_grad()
def validate() -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for x, y in tqdm(valid_loader, desc="valid", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += x.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)
"""
            ),
            md("## 5. Train And Save Artifacts"),
            md(
                """
The notebook saves only the best validation checkpoint. `history.csv` records the learning curve so later experiments can compare changes without relying on notebook output logs.
"""
            ),
            code(
                """
history = []
best_acc = 0.0

for epoch in range(1, CFG.epochs + 1):
    train_loss = train_one_epoch()
    valid_loss, valid_acc = validate()
    scheduler.step()
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "lr": scheduler.get_last_lr()[0],
    }
    history.append(row)
    print(row)

    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(
            {
                "model": model.state_dict(),
                "label_to_idx": label_to_idx,
                "cfg": {k: v for k, v in CFG.__dict__.items() if not k.startswith("_")},
                "valid_acc": best_acc,
            },
            CFG.artifact_dir / "best_effnet_b0.pt",
        )

history_df = pd.DataFrame(history)
history_df.to_csv(CFG.artifact_dir / "history.csv", index=False)
print(f"Best valid accuracy: {best_acc:.4f}")
print(f"Artifacts saved to {CFG.artifact_dir}")
"""
            ),
            md("## 6. Package Artifacts For Download"),
            md(
                """
Kaggle exposes files under `/kaggle/working` after the run. This cell packages the model checkpoint, label map, training history, and fold file into one zip for local download or reuse in an inference notebook.
"""
            ),
            code(
                """
import zipfile
from IPython.display import FileLink

zip_path = Path("/kaggle/working/birdclef_effnet_b0_artifacts.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(CFG.artifact_dir.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=path.relative_to(CFG.artifact_dir.parent))

print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")
display(FileLink(zip_path))
"""
            ),
        ]
    )


def perch_notebook() -> dict:
    return notebook(
        [
            md(
                """
# BirdCLEF+ 2026 - Google Perch v2 Probe

**Purpose.** Use Google Perch v2 as a frozen bioacoustic feature extractor and train a lightweight PyTorch probe on 1,536-dimensional embeddings.

**Run mode.** Kaggle training/diagnostic notebook. It may require TensorFlow 2.20 wheels for the current Perch export.

**Primary outputs.** Perch embeddings, probe checkpoint, label mapping, training history, and a zipped artifact bundle.

**Submission note.** Direct Perch inference is too slow for hidden competition reruns in our current setup. Use Perch as a teacher/offline feature extractor, then submit a fast PyTorch student model.

Artifacts are written to `/kaggle/working/artifacts/perch_v2`.
"""
            ),
            md("## 1. Setup"),
            code(
                COMMON_STYLE
                + """
import librosa
import subprocess
from importlib.metadata import PackageNotFoundError, version
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

REQUIRED_TENSORFLOW_VERSION = "2.20.0"
OFFLINE_TENSORFLOW_WHEELHOUSE = Path("/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel")


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def version_tuple(value: str) -> tuple[int, ...]:
    core = value.split("+")[0].split("-")[0]
    parts = []
    for part in core.split("."):
        if part.isdigit():
            parts.append(int(part))
        else:
            break
    return tuple(parts)


installed_tf = package_version("tensorflow")
if installed_tf is None or version_tuple(installed_tf) < version_tuple(REQUIRED_TENSORFLOW_VERSION):
    import sys
    print(f"TensorFlow {installed_tf} is not compatible with Perch v2 export 2.")
    print(f"Installing tensorflow=={REQUIRED_TENSORFLOW_VERSION} before importing TensorFlow.")
    if OFFLINE_TENSORFLOW_WHEELHOUSE.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                str(OFFLINE_TENSORFLOW_WHEELHOUSE / "tensorboard-2.20.0-py3-none-any.whl"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                str(
                    OFFLINE_TENSORFLOW_WHEELHOUSE
                    / "tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
                ),
            ],
            check=True,
        )
        print("Installed TF 2.20.0 from Kaggle dataset wheels.")
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "tensorflow==2.20.0", "tensorboard==2.20.0"],
            check=True,
        )
        print("Installed TF 2.20.0 from PyPI.")
    installed_tf = package_version("tensorflow")
    if installed_tf is None or version_tuple(installed_tf) < version_tuple(REQUIRED_TENSORFLOW_VERSION):
        raise RuntimeError(
            "TensorFlow upgrade did not complete. If Kaggle shows DNS or connection errors, enable Internet "
            "for this notebook or attach an offline TensorFlow wheelhouse dataset and set "
            "OFFLINE_TENSORFLOW_WHEELHOUSE to that /kaggle/input path."
        )
    print(f"TensorFlow package is now {installed_tf}; importing TensorFlow next.")

try:
    import tensorflow as tf
except ImportError:
    tf = None

torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)


class CFG(CFG):
    artifact_dir = Path("/kaggle/working/artifacts/perch_v2")
    sample_rate = 32000
    duration = 5.0
    embedding_dim = 1536
    extraction_batch_size = 8
    probe_batch_size = 128
    probe_epochs = 10
    hidden_dim = 512
    dropout = 0.25
    lr = 1e-3
    max_samples = None
    perch_model_dir = None


CFG.artifact_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if tf is not None:
    print(f"TensorFlow: {tf.__version__}")
"""
            ),
            md("## 2. Load Metadata"),
            md(
                """
The probe is trained on the same 206 primary labels as the EfficientNet baseline. This keeps the comparison clean: the main difference is the representation, not the target definition.
"""
            ),
            code(
                """
train = pd.read_csv(CFG.data_root / "train.csv")
train["filepath"] = train["filename"].map(lambda x: CFG.data_root / "train_audio" / x)
if CFG.max_samples:
    train = train.sample(CFG.max_samples, random_state=CFG.seed).reset_index(drop=True)

labels = sorted(train["primary_label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
train["target"] = train["primary_label"].map(label_to_idx)
(CFG.artifact_dir / "labels.json").write_text(json.dumps(idx_to_label, indent=2), encoding="utf-8")

print(f"Rows: {len(train):,}")
print(f"Classes: {len(labels):,}")
display(train.head())
"""
            ),
            md("## 3. Locate And Load Perch"),
            md(
                """
Perch v2 is loaded as a TensorFlow SavedModel. The notebook performs a smoke test immediately after loading so TensorFlow/XLA compatibility issues surface before the long embedding extraction loop starts.
"""
            ),
            code(
                """
def find_perch_model_dir() -> Path:
    if CFG.perch_model_dir:
        return Path(CFG.perch_model_dir)
    input_root = Path("/kaggle/input")
    matches = list(input_root.glob("**/saved_model.pb")) if input_root.exists() else []
    matches = [path.parent for path in matches if "perch" in str(path).lower() or "vocal" in str(path).lower()]
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find a Perch SavedModel. Attach the Perch/vocalization-classifier Kaggle model "
        "or set CFG.perch_model_dir."
    )


if tf is None:
    raise ImportError("TensorFlow is required for Perch embedding extraction.")

perch_model_dir = find_perch_model_dir()
perch = tf.saved_model.load(str(perch_model_dir))
infer = perch.signatures["serving_default"]
print(f"Perch model: {perch_model_dir}")
print(f"Inputs: {infer.structured_input_signature}")
print(f"Outputs: {infer.structured_outputs}")


def explain_perch_runtime_error(error: Exception) -> None:
    message = str(error)
    if "XlaCallModuleOp with version 10 is not supported" in message:
        raise RuntimeError(
            "The attached Perch SavedModel requires a newer TensorFlow/XLA runtime than this Kaggle image. "
            "The setup cell should install tensorflow>=2.20 before TensorFlow is imported. Restart the "
            "Kaggle session after installation, then run the notebook from the top. If internet is disabled, "
            "attach a Kaggle dataset containing a compatible TensorFlow wheel or use an older Perch SavedModel export."
        ) from error
    raise error


def smoke_test_perch() -> None:
    dummy = tf.zeros((1, int(CFG.sample_rate * CFG.duration)), dtype=tf.float32)
    _, keyword_specs = infer.structured_input_signature
    try:
        if keyword_specs:
            input_name = next(iter(keyword_specs))
            outputs = infer(**{input_name: dummy})
        else:
            outputs = infer(dummy)
    except Exception as error:
        explain_perch_runtime_error(error)
    print({name: tuple(value.shape) for name, value in outputs.items()})


smoke_test_perch()
"""
            ),
            md("## 4. Extract Embeddings"),
            md(
                """
Embeddings are cached to disk because TensorFlow Perch extraction is the expensive step. Once `train_embeddings.npz` exists, the PyTorch probe can be retrained quickly without rerunning the backbone.
"""
            ),
            code(
                """
def load_audio(path: Path) -> np.ndarray:
    target_len = int(CFG.sample_rate * CFG.duration)
    y, _ = librosa.load(path, sr=CFG.sample_rate, mono=True, duration=CFG.duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len].astype(np.float32)


def run_perch_batch(batch_waveforms: np.ndarray) -> np.ndarray:
    tensor = tf.convert_to_tensor(batch_waveforms, dtype=tf.float32)
    _, keyword_specs = infer.structured_input_signature
    if keyword_specs:
        input_name = next(iter(keyword_specs))
        try:
            outputs = infer(**{input_name: tensor})
        except Exception as error:
            explain_perch_runtime_error(error)
    else:
        try:
            outputs = infer(tensor)
        except Exception as error:
            explain_perch_runtime_error(error)
    arrays = {name: np.asarray(value) for name, value in outputs.items()}
    embedding_name = next(
        (name for name in arrays if "embedding" in name.lower() or "embed" in name.lower()),
        next(iter(arrays)),
    )
    value = arrays[embedding_name]
    if value.ndim == 3:
        value = value.mean(axis=1)
    elif value.ndim > 3:
        value = value.reshape(value.shape[0], -1)
    return value.astype(np.float32)


embeddings_path = CFG.artifact_dir / "train_embeddings.npz"
if embeddings_path.exists():
    saved = np.load(embeddings_path, allow_pickle=True)
    embeddings = saved["embeddings"]
else:
    chunks = []
    waveforms = []
    for path in tqdm(train["filepath"], desc="audio"):
        waveforms.append(load_audio(path))
        if len(waveforms) == CFG.extraction_batch_size:
            chunks.append(run_perch_batch(np.stack(waveforms)))
            waveforms = []
    if waveforms:
        chunks.append(run_perch_batch(np.stack(waveforms)))
    embeddings = np.concatenate(chunks, axis=0)
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        labels=train["primary_label"].to_numpy(),
        filenames=train["filename"].to_numpy(),
    )

print(f"Embeddings: {embeddings.shape}")
print(f"Saved: {embeddings_path}")
"""
            ),
            md("## 5. Probe Model"),
            md(
                """
The probe is deliberately shallow. Perch already encodes strong acoustic features, so the probe should learn a compact decision boundary rather than memorize the training set.
"""
            ),
            code(
                """
class PerchProbe(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, CFG.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


x = embeddings.astype(np.float32)
y = train["target"].to_numpy(dtype=np.int64)


def safe_train_valid_split(targets: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    counts = pd.Series(targets).value_counts()
    rare_classes = set(counts[counts < 2].index)
    all_idx = np.arange(len(targets))
    rare_idx = np.array([idx for idx in all_idx if targets[idx] in rare_classes], dtype=np.int64)
    common_idx = np.array([idx for idx in all_idx if targets[idx] not in rare_classes], dtype=np.int64)

    train_common, valid_idx = train_test_split(
        common_idx,
        test_size=test_size,
        random_state=CFG.seed,
        stratify=targets[common_idx],
    )
    train_idx = np.concatenate([train_common, rare_idx])
    return train_idx, valid_idx


train_idx, valid_idx = safe_train_valid_split(y)
print(f"Probe train rows: {len(train_idx):,}")
print(f"Probe valid rows: {len(valid_idx):,}")
print(f"Classes with fewer than 2 rows kept in train only: {(pd.Series(y).value_counts() < 2).sum():,}")

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
    batch_size=CFG.probe_batch_size,
    shuffle=True,
)
valid_loader = DataLoader(
    TensorDataset(torch.from_numpy(x[valid_idx]), torch.from_numpy(y[valid_idx])),
    batch_size=CFG.probe_batch_size * 2,
    shuffle=False,
)

model = PerchProbe(embedding_dim=x.shape[1], num_classes=len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
"""
            ),
            md("## 6. Train And Save Artifacts"),
            md(
                """
Singleton classes are kept in the training split to avoid invalid stratification. Validation accuracy should be interpreted as a directional signal rather than a perfect proxy for hidden soundscape performance.
"""
            ),
            code(
                """
@torch.no_grad()
def validate() -> float:
    model.eval()
    correct = 0
    seen = 0
    for xb, yb in valid_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        seen += xb.size(0)
    return correct / max(seen, 1)


history = []
best_acc = 0.0
for epoch in range(1, CFG.probe_epochs + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    valid_acc = validate()
    row = {
        "epoch": epoch,
        "train_loss": total_loss / max(len(train_loader.dataset), 1),
        "valid_acc": valid_acc,
    }
    history.append(row)
    print(row)
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(
            {
                "model": model.state_dict(),
                "label_to_idx": label_to_idx,
                "cfg": {k: v for k, v in CFG.__dict__.items() if not k.startswith("_")},
                "valid_acc": best_acc,
            },
            CFG.artifact_dir / "best_perch_probe.pt",
        )

pd.DataFrame(history).to_csv(CFG.artifact_dir / "history.csv", index=False)
print(f"Best valid accuracy: {best_acc:.4f}")
print(f"Artifacts saved to {CFG.artifact_dir}")
"""
            ),
            md("## 7. Package Artifacts For Download"),
            md(
                """
This cell zips the extracted Perch embeddings, label map, probe checkpoint, and training history. The embeddings file can be large, but keeping it in the zip is useful when you want to train probes locally without rerunning TensorFlow extraction.
"""
            ),
            code(
                """
import zipfile
from IPython.display import FileLink

zip_path = Path("/kaggle/working/birdclef_perch_v2_artifacts.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(CFG.artifact_dir.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=path.relative_to(CFG.artifact_dir.parent))

print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")
display(FileLink(zip_path))
"""
            ),
        ]
    )


def submission_notebook() -> dict:
    return notebook(
        [
            md(
                """
# BirdCLEF+ 2026 - EfficientNet-B0 Submission

**Purpose.** Generate a competition-safe `submission.csv` from the trained EfficientNet-B0 baseline.

**Run mode.** Kaggle submission notebook. This is the reliable official-submission path because it uses only PyTorch and fast mel-spectrogram inference.

**Required inputs.** BirdCLEF+ 2026 competition data and the uploaded EfficientNet artifact containing `best_effnet_b0.pt` and `labels.json`.

**Primary outputs.** `/kaggle/working/submission.csv` and a zipped submission artifact bundle.

Artifacts are written to `/kaggle/working/artifacts/submission`.
"""
            ),
            md("## 1. Setup"),
            code(
                COMMON_STYLE
                + """
try:
    import timm
except ImportError:
    import sys
    !{sys.executable} -m pip install -q timm
    import timm

import librosa
import zipfile
import torch
from torch import nn
from tqdm.auto import tqdm
from IPython.display import FileLink

torch.set_num_threads(min(2, os.cpu_count() or 1))


class CFG(CFG):
    artifact_dir = Path("/kaggle/working/artifacts/submission")
    sample_rate = 32000
    duration = 5.0
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20
    fmax = 16000
    backbone = "efficientnet_b0"
    batch_size = 32
    checkpoint_path = None
    labels_path = None


CFG.artifact_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
"""
            ),
            md("## 2. Locate Competition Files And Model Artifact"),
            md(
                """
The notebook searches attached Kaggle inputs for the trained checkpoint and label map. Keeping artifact discovery flexible makes the same notebook work with either GitHub files or Kaggle Model uploads.
"""
            ),
            code(
                """
def find_file(filename: str, roots: list[Path]) -> Path:
    for root in roots:
        if not root.exists():
            continue
        direct = root / filename
        if direct.exists():
            return direct
        matches = list(root.glob(f"**/{filename}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {filename}. Attach the EfficientNet artifact dataset.")


artifact_roots = [
    Path("/kaggle/input"),
    Path("/kaggle/working/artifacts/effnet_b0"),
    Path("artifacts/effnet_b0"),
]
sample_path = CFG.data_root / "sample_submission.csv"
checkpoint_path = Path(CFG.checkpoint_path) if CFG.checkpoint_path else find_file("best_effnet_b0.pt", artifact_roots)
labels_path = Path(CFG.labels_path) if CFG.labels_path else find_file("labels.json", artifact_roots)

sample_submission = pd.read_csv(sample_path)
idx_to_label = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}
labels = [idx_to_label[i] for i in sorted(idx_to_label)]
label_to_idx = {label: i for i, label in enumerate(labels)}
target_columns = [c for c in sample_submission.columns if c != "row_id"]

print(f"Sample submission: {sample_path}")
print(f"Checkpoint: {checkpoint_path}")
print(f"Labels: {labels_path}")
print(f"Rows: {len(sample_submission):,}")
print(f"Submission target columns: {len(target_columns):,}")
print(f"Model classes: {len(labels):,}")
display(sample_submission.head())
"""
            ),
            md("## 3. Build Model And Audio Helpers"),
            code(
                """
class BirdClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = timm.create_model(
            CFG.backbone,
            pretrained=False,
            in_chans=1,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


checkpoint = torch_load(checkpoint_path)
model = BirdClassifier(num_classes=len(labels)).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()


def row_to_stem_and_end_time(row_id: str) -> tuple[str, float]:
    stem, sep, end = str(row_id).rpartition("_")
    if sep and end.replace(".", "", 1).isdigit():
        return stem, float(end)
    return str(row_id), CFG.duration


def build_audio_index() -> dict[str, Path]:
    candidates = [
        CFG.data_root / "test_soundscapes",
        CFG.data_root / "test_audio",
        CFG.data_root / "train_soundscapes",
    ]
    audio_index = {}
    for folder in candidates:
        if folder.exists():
            for ext in ("*.ogg", "*.wav", "*.flac", "*.mp3"):
                for path in folder.glob(ext):
                    audio_index[path.stem] = path
    return audio_index


audio_index = build_audio_index()
print(f"Indexed audio files: {len(audio_index):,}")
"""
            ),
            md("## 4. Run Inference"),
            md(
                """
Each row is interpreted as a 5-second window ending at the timestamp encoded in `row_id`. The model predicts probabilities for every label column expected by `sample_submission.csv`.
"""
            ),
            code(
                """
def load_audio_segment(path: Path, end_time: float) -> np.ndarray:
    offset = max(0.0, float(end_time) - CFG.duration)
    target_len = int(CFG.sample_rate * CFG.duration)
    y, _ = librosa.load(path, sr=CFG.sample_rate, mono=True, offset=offset, duration=CFG.duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len].astype(np.float32)


def audio_to_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=CFG.sample_rate,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        n_mels=CFG.n_mels,
        fmin=CFG.fmin,
        fmax=CFG.fmax,
        power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.astype(np.float32)


def predict_batch(batch: list[np.ndarray]) -> np.ndarray:
    x = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


submission = sample_submission.copy()
for col in target_columns:
    submission[col] = 0.0

batch = []
batch_rows = []
missing_audio = []

for row_idx, row_id in tqdm(list(enumerate(submission["row_id"])), desc="inference"):
    stem, end_time = row_to_stem_and_end_time(row_id)
    audio_path = audio_index.get(stem)
    if audio_path is None:
        missing_audio.append(row_id)
        continue
    batch.append(audio_to_mel(load_audio_segment(audio_path, end_time)))
    batch_rows.append(row_idx)
    if len(batch) == CFG.batch_size:
        probs = predict_batch(batch)
        for local_idx, submit_idx in enumerate(batch_rows):
            for label, class_idx in label_to_idx.items():
                if label in target_columns:
                    submission.loc[submit_idx, label] = probs[local_idx, class_idx]
        batch = []
        batch_rows = []

if batch:
    probs = predict_batch(batch)
    for local_idx, submit_idx in enumerate(batch_rows):
        for label, class_idx in label_to_idx.items():
            if label in target_columns:
                submission.loc[submit_idx, label] = probs[local_idx, class_idx]

missing_audio_path = CFG.artifact_dir / "missing_test_audio.json"
missing_audio_path.write_text(json.dumps(missing_audio, indent=2), encoding="utf-8")
print(f"Missing audio rows: {len(missing_audio):,}")
display(submission.head())
"""
            ),
            md("## 5. Save Submission And Package Artifacts"),
            code(
                """
submission_path = Path("/kaggle/working/submission.csv")
submission.to_csv(submission_path, index=False)
submission.to_csv(CFG.artifact_dir / "submission.csv", index=False)

zip_path = Path("/kaggle/working/birdclef_effnet_b0_submission_artifacts.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(submission_path, arcname="submission.csv")
    for path in sorted(CFG.artifact_dir.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=path.relative_to(CFG.artifact_dir.parent))

print(f"Wrote submission: {submission_path}")
print(f"Wrote artifact zip: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")
display(FileLink(submission_path))
display(FileLink(zip_path))
"""
            ),
        ]
    )


def perch_submission_notebook() -> dict:
    return notebook(
        [
            md(
                """
# BirdCLEF+ 2026 - Perch v2 Probe Submission

**Purpose.** Generate a Perch-probe submission when compatible cached embeddings or a CPU-capable Perch model are available.

**Run mode.** Experimental/diagnostic submission notebook. Direct Perch inference can exceed hidden rerun limits, so this is not the default official-submission path.

**Important.** For CPU competition reruns, attach cached embeddings that cover the submission row IDs or attach a `perch_v2_cpu` model dataset. The standard `perch_v2/2` SavedModel is CUDA-only and cannot run on CPU.

**Recommended official path.** Use `04_effnet_b0_submission.ipynb` unless this notebook confirms cache coverage or a compatible CPU Perch model.

Artifacts are written to `/kaggle/working/artifacts/perch_submission`.
"""
            ),
            md("## 1. Setup"),
            code(
                COMMON_STYLE
                + """
import zipfile
import subprocess
from importlib.metadata import PackageNotFoundError, version

import librosa
import torch
from torch import nn
from tqdm.auto import tqdm
from IPython.display import FileLink

REQUIRED_TENSORFLOW_VERSION = "2.20.0"
OFFLINE_TENSORFLOW_WHEELHOUSE = Path("/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel")


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def version_tuple(value: str) -> tuple[int, ...]:
    core = value.split("+")[0].split("-")[0]
    parts = []
    for part in core.split("."):
        if part.isdigit():
            parts.append(int(part))
        else:
            break
    return tuple(parts)


installed_tf = package_version("tensorflow")
if installed_tf is None or version_tuple(installed_tf) < version_tuple(REQUIRED_TENSORFLOW_VERSION):
    import sys
    print(f"TensorFlow {installed_tf} is not compatible with Perch v2 export 2.")
    print(f"Installing tensorflow=={REQUIRED_TENSORFLOW_VERSION} before importing TensorFlow.")
    if OFFLINE_TENSORFLOW_WHEELHOUSE.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                str(OFFLINE_TENSORFLOW_WHEELHOUSE / "tensorboard-2.20.0-py3-none-any.whl"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                str(
                    OFFLINE_TENSORFLOW_WHEELHOUSE
                    / "tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
                ),
            ],
            check=True,
        )
        print("Installed TF 2.20.0 from Kaggle dataset wheels.")
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "tensorflow==2.20.0", "tensorboard==2.20.0"],
            check=True,
        )
        print("Installed TF 2.20.0 from PyPI.")
    installed_tf = package_version("tensorflow")
    if installed_tf is None or version_tuple(installed_tf) < version_tuple(REQUIRED_TENSORFLOW_VERSION):
        raise RuntimeError(
            "TensorFlow upgrade did not complete. If Kaggle shows DNS or connection errors, enable Internet "
            "for this notebook or attach an offline TensorFlow wheelhouse dataset and set "
            "OFFLINE_TENSORFLOW_WHEELHOUSE to that /kaggle/input path."
        )
    print(f"TensorFlow package is now {installed_tf}; importing TensorFlow next.")

import tensorflow as tf

torch.set_num_threads(min(2, os.cpu_count() or 1))


class CFG(CFG):
    artifact_dir = Path("/kaggle/working/artifacts/perch_submission")
    sample_rate = 32000
    duration = 5.0
    extraction_batch_size = 8
    probe_batch_size = 256
    hidden_dim = 512
    dropout = 0.25
    perch_model_dir = None
    probe_checkpoint_path = None
    labels_path = None
    require_cpu_perch_for_cpu_session = True


CFG.artifact_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"TensorFlow: {tf.__version__}")
"""
            ),
            md("## 2. Locate Competition Files And Model Artifacts"),
            md(
                """
This section first looks for cached Perch embeddings. If the cache covers all requested `row_id`s, the notebook skips TensorFlow entirely and runs only the PyTorch probe. If the cache does not cover the submission rows, a compatible Perch SavedModel is required.
"""
            ),
            code(
                """
def find_file(filename: str, roots: list[Path]) -> Path:
    for root in roots:
        if not root.exists():
            continue
        direct = root / filename
        if direct.exists():
            return direct
        matches = list(root.glob(f"**/{filename}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {filename}. Attach the Perch probe model artifact.")


def find_perch_model_dir() -> Path:
    if CFG.perch_model_dir:
        return Path(CFG.perch_model_dir)
    input_root = Path("/kaggle/input")
    matches = list(input_root.glob("**/saved_model.pb")) if input_root.exists() else []
    perch_matches = [path.parent for path in matches if "perch" in str(path).lower() or "vocal" in str(path).lower()]
    cpu_matches = [path for path in perch_matches if "perch_v2_cpu" in str(path).lower() or "cpu" in str(path).lower()]
    cuda_matches = [path for path in perch_matches if path not in cpu_matches]
    if device.type == "cpu":
        if cpu_matches:
            return cpu_matches[0]
        if CFG.require_cpu_perch_for_cpu_session:
            raise FileNotFoundError(
                "CPU session detected, but no perch_v2_cpu SavedModel was found. "
                "Attach a Kaggle model/dataset containing perch_v2_cpu. The standard perch_v2/2 export is CUDA-only."
            )
    if cpu_matches:
        return cpu_matches[0]
    if cuda_matches:
        return cuda_matches[0]
    raise FileNotFoundError("Could not find Google Perch v2 SavedModel. Attach the Kaggle Perch model.")


def find_perch_cache() -> tuple[Path | None, Path | None]:
    candidates = [
        Path("/kaggle/input/perch-meta"),
        Path("/kaggle/input/datasets/jaejohn/perch-meta"),
        Path("/kaggle/working/perch_cache"),
    ]
    for root in candidates:
        meta_path = root / "full_perch_meta.parquet"
        arrays_path = root / "full_perch_arrays.npz"
        if meta_path.exists() and arrays_path.exists():
            return meta_path, arrays_path
    input_root = Path("/kaggle/input")
    if input_root.exists():
        meta_matches = list(input_root.glob("**/full_perch_meta.parquet"))
        for meta_path in meta_matches:
            arrays_path = meta_path.parent / "full_perch_arrays.npz"
            if arrays_path.exists():
                return meta_path, arrays_path
    return None, None


artifact_roots = [Path("/kaggle/input"), Path("artifacts/perch_v2")]
sample_path = CFG.data_root / "sample_submission.csv"
probe_checkpoint_path = Path(CFG.probe_checkpoint_path) if CFG.probe_checkpoint_path else find_file("best_perch_probe.pt", artifact_roots)
labels_path = Path(CFG.labels_path) if CFG.labels_path else find_file("labels.json", artifact_roots)
sample_submission = pd.read_csv(sample_path)
idx_to_label = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}
labels = [idx_to_label[i] for i in sorted(idx_to_label)]
label_to_idx = {label: i for i, label in enumerate(labels)}
target_columns = [c for c in sample_submission.columns if c != "row_id"]
test_soundscape_files = sorted((CFG.data_root / "test_soundscapes").glob("*.ogg"))
use_sample_submission_only = len(test_soundscape_files) == 0 and len(sample_submission) <= 3

cache_meta_path, cache_arrays_path = find_perch_cache()
use_cached_embeddings = False
if cache_meta_path is not None and cache_arrays_path is not None:
    cache_meta_preview = pd.read_parquet(cache_meta_path, columns=["row_id"])
    cache_row_ids = set(cache_meta_preview["row_id"].astype(str))
    requested_row_ids = set(sample_submission["row_id"].astype(str))
    use_cached_embeddings = requested_row_ids.issubset(cache_row_ids)
    print(f"Perch cache: {cache_meta_path}")
    print(f"Cache rows: {len(cache_row_ids):,}; requested rows: {len(requested_row_ids):,}; covers submission: {use_cached_embeddings}")

perch_model_dir = None if (use_cached_embeddings or use_sample_submission_only) else find_perch_model_dir()

print(f"Sample submission: {sample_path}")
if use_sample_submission_only:
    print("Public dry-run detected: no test soundscapes and sample_submission has <= 3 rows.")
    print("Perch model: not needed; writing sample_submission.csv unchanged.")
else:
    print(f"Perch model: {perch_model_dir if perch_model_dir is not None else 'not needed; using cached embeddings'}")
print(f"Probe checkpoint: {probe_checkpoint_path}")
print(f"Labels: {labels_path}")
print(f"Rows: {len(sample_submission):,}")
print(f"Submission target columns: {len(target_columns):,}")
print(f"Probe classes: {len(labels):,}")
display(sample_submission.head())
"""
            ),
            md("## 3. Load Perch And Probe"),
            md(
                """
When cache coverage is available, this section only initializes the probe shape. Otherwise it loads Perch and runs a smoke test before any full inference loop begins.
"""
            ),
            code(
                """
cache_meta = None
cache_embeddings = None
cache_row_to_idx = None

if use_sample_submission_only:
    dummy_embedding = np.zeros((1, 1536), dtype=np.float32)
    print("Skipping Perch/probe loading for public dry-run.")
elif use_cached_embeddings:
    cache_meta = pd.read_parquet(cache_meta_path)
    cache_arrays = np.load(cache_arrays_path)
    cache_embeddings = cache_arrays["emb_full"].astype(np.float32)
    cache_row_to_idx = {str(row_id): i for i, row_id in enumerate(cache_meta["row_id"].astype(str))}
    dummy_embedding = cache_embeddings[:1]
    print(f"Loaded cached Perch embeddings: {cache_embeddings.shape}")
else:
    perch = tf.saved_model.load(str(perch_model_dir))
    infer = perch.signatures["serving_default"]
    print(f"Inputs: {infer.structured_input_signature}")
    print(f"Outputs: {infer.structured_outputs}")


def explain_perch_runtime_error(error: Exception) -> None:
    message = str(error)
    if "XlaCallModuleOp with version 10 is not supported" in message:
        raise RuntimeError(
            "This Perch SavedModel requires TensorFlow/XLA >= the version installed in this session. "
            "The setup cell should install tensorflow>=2.20 before import. Restart the Kaggle session "
            "after installation, then run from the top."
        ) from error
    if "current platform CPU is not among the platforms required" in message:
        raise RuntimeError(
            "The selected Perch SavedModel is CUDA-only, but this session is running on CPU. "
            "Attach/use a perch_v2_cpu model dataset for CPU submission, or switch to the EfficientNet submission."
        ) from error
    raise error


def run_perch_batch(batch_waveforms: np.ndarray) -> np.ndarray:
    if use_cached_embeddings:
        raise RuntimeError("run_perch_batch should not be called when cached embeddings are active.")
    tensor = tf.convert_to_tensor(batch_waveforms, dtype=tf.float32)
    _, keyword_specs = infer.structured_input_signature
    try:
        if keyword_specs:
            input_name = next(iter(keyword_specs))
            outputs = infer(**{input_name: tensor})
        else:
            outputs = infer(tensor)
    except Exception as error:
        explain_perch_runtime_error(error)

    arrays = {name: np.asarray(value) for name, value in outputs.items()}
    embedding_name = next(
        (name for name in arrays if "embedding" in name.lower() and "spatial" not in name.lower()),
        next(iter(arrays)),
    )
    value = arrays[embedding_name]
    if value.ndim == 3:
        value = value.mean(axis=1)
    elif value.ndim > 3:
        value = value.reshape(value.shape[0], -1)
    return value.astype(np.float32)


if not use_cached_embeddings and not use_sample_submission_only:
    dummy = np.zeros((1, int(CFG.sample_rate * CFG.duration)), dtype=np.float32)
    dummy_embedding = run_perch_batch(dummy)
print(f"Embedding shape: {dummy_embedding.shape}")


class PerchProbe(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, CFG.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


probe = PerchProbe(embedding_dim=dummy_embedding.shape[1], num_classes=len(labels)).to(device)
if not use_sample_submission_only:
    checkpoint = torch_load(probe_checkpoint_path)
    if "model" in checkpoint:
        probe.load_state_dict(checkpoint["model"])
    else:
        probe.load_state_dict(checkpoint)
probe.eval()
"""
            ),
            md("## 4. Build Test Audio Index"),
            code(
                """
def row_to_stem_and_end_time(row_id: str) -> tuple[str, float]:
    stem, sep, end = str(row_id).rpartition("_")
    if sep and end.replace(".", "", 1).isdigit():
        return stem, float(end)
    return str(row_id), CFG.duration


def build_audio_index() -> dict[str, Path]:
    candidates = [
        CFG.data_root / "test_soundscapes",
        CFG.data_root / "test_audio",
        CFG.data_root / "train_soundscapes",
    ]
    audio_index = {}
    for folder in candidates:
        if folder.exists():
            for ext in ("*.ogg", "*.wav", "*.flac", "*.mp3"):
                for path in folder.glob(ext):
                    audio_index[path.stem] = path
    return audio_index


def load_audio_segment(path: Path, end_time: float) -> np.ndarray:
    offset = max(0.0, float(end_time) - CFG.duration)
    target_len = int(CFG.sample_rate * CFG.duration)
    y, _ = librosa.load(path, sr=CFG.sample_rate, mono=True, offset=offset, duration=CFG.duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len].astype(np.float32)


audio_index = build_audio_index()
print(f"Indexed audio files: {len(audio_index):,}")
"""
            ),
            md("## 5. Run Perch Probe Inference"),
            md(
                """
The fast path uses cached embeddings. The slow path extracts Perch embeddings from audio and should be treated as diagnostic unless runtime has been confirmed against the hidden rerun budget.
"""
            ),
            code(
                """
def predict_probe(embeddings: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(embeddings.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = probe(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


submission = sample_submission.copy()
for col in target_columns:
    submission[col] = 0.0

waveforms = []
batch_rows = []
missing_audio = []

if use_sample_submission_only:
    submission = sample_submission.copy()
elif use_cached_embeddings:
    cached = cache_embeddings[[cache_row_to_idx[str(row_id)] for row_id in submission["row_id"].astype(str)]]
    for start in tqdm(range(0, len(submission), CFG.probe_batch_size), desc="cached probe"):
        end = min(start + CFG.probe_batch_size, len(submission))
        probs = predict_probe(cached[start:end])
        for local_idx, submit_idx in enumerate(range(start, end)):
            for label, class_idx in label_to_idx.items():
                if label in target_columns:
                    submission.loc[submit_idx, label] = probs[local_idx, class_idx]
else:
    for row_idx, row_id in tqdm(list(enumerate(submission["row_id"])), desc="perch submission"):
        stem, end_time = row_to_stem_and_end_time(row_id)
        audio_path = audio_index.get(stem)
        if audio_path is None:
            missing_audio.append(row_id)
            continue

        waveforms.append(load_audio_segment(audio_path, end_time))
        batch_rows.append(row_idx)

        if len(waveforms) == CFG.extraction_batch_size:
            embeddings = run_perch_batch(np.stack(waveforms))
            probs = predict_probe(embeddings)
            for local_idx, submit_idx in enumerate(batch_rows):
                for label, class_idx in label_to_idx.items():
                    if label in target_columns:
                        submission.loc[submit_idx, label] = probs[local_idx, class_idx]
            waveforms = []
            batch_rows = []

    if waveforms:
        embeddings = run_perch_batch(np.stack(waveforms))
        probs = predict_probe(embeddings)
        for local_idx, submit_idx in enumerate(batch_rows):
            for label, class_idx in label_to_idx.items():
                if label in target_columns:
                    submission.loc[submit_idx, label] = probs[local_idx, class_idx]

"""
            ),
            md("## 6. Save Submission And Package Artifacts"),
            code(
                """
missing_audio_path = CFG.artifact_dir / "missing_test_audio.json"
missing_audio_path.write_text(json.dumps(missing_audio, indent=2), encoding="utf-8")
print(f"Missing audio rows: {len(missing_audio):,}")
display(submission.head())
"""
            ),
            code(
                """
submission_path = Path("/kaggle/working/submission.csv")
submission.to_csv(submission_path, index=False)
submission.to_csv(CFG.artifact_dir / "submission.csv", index=False)

zip_path = Path("/kaggle/working/birdclef_perch_v2_submission_artifacts.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(submission_path, arcname="submission.csv")
    for path in sorted(CFG.artifact_dir.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=path.relative_to(CFG.artifact_dir.parent))

print(f"Wrote submission: {submission_path}")
print(f"Wrote artifact zip: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")
display(FileLink(submission_path))
display(FileLink(zip_path))
"""
            ),
        ]
    )

def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_data_eda.ipynb": eda_notebook(),
        "02_effnet_b0_baseline.ipynb": effnet_notebook(),
        "03_perch_v2_probe.ipynb": perch_notebook(),
        "04_effnet_b0_submission.ipynb": submission_notebook(),
        "05_perch_v2_submission.ipynb": perch_submission_notebook(),
    }
    for name, nb in outputs.items():
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
