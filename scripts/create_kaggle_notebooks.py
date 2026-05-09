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

Purpose: understand the dataset shape, label imbalance, taxonomy coverage, duration/chunking strategy, secondary-label noise, soundscape annotations, and representative audio examples.

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
            md("## 4. Primary Label Imbalance"),
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
            md("## 5. Duration And Chunking Implications"),
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
            md("## 6. Secondary Labels And Co-Occurrence"),
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
            md("## 7. Taxonomy Coverage"),
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
            md("## 8. Soundscape Labels"),
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
    display(soundscape_labels.head())
    print(soundscape_labels.columns.tolist())

    label_like_cols = [c for c in soundscape_labels.columns if "label" in c.lower() or "species" in c.lower() or "code" in c.lower()]
    time_like_cols = [c for c in soundscape_labels.columns if "time" in c.lower() or "second" in c.lower()]
    file_like_cols = [c for c in soundscape_labels.columns if "filename" in c.lower() or "soundscape" in c.lower() or "row_id" in c.lower()]

    soundscape_summary = {
        "rows": len(soundscape_labels),
        "columns": len(soundscape_labels.columns),
        "label_like_columns": ", ".join(label_like_cols),
        "time_like_columns": ", ".join(time_like_cols),
        "file_like_columns": ", ".join(file_like_cols),
    }
    pd.Series(soundscape_summary).to_csv(CFG.artifact_dir / "soundscape_summary.csv", header=["value"])
    display(pd.Series(soundscape_summary).to_frame("value"))

    if label_like_cols:
        col = label_like_cols[0]
        sc_counts = soundscape_labels[col].value_counts().head(CFG.top_n).rename_axis(col).reset_index(name="rows")
        sc_counts.to_csv(CFG.artifact_dir / "soundscape_label_counts.csv", index=False)
        display(sc_counts)

"""
            ),
            md("## 9. Representative Audio And Spectrograms"),
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
            md("## 10. Modeling Takeaways"),
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
            md("## 11. Artifact Manifest"),
            code(
                """
manifest = sorted(str(path.relative_to(CFG.artifact_dir)) for path in CFG.artifact_dir.glob("*"))
(CFG.artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
manifest
"""
            ),
            md("## 12. Package Artifacts For Download"),
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

Purpose: train a reproducible mel-spectrogram EfficientNet-B0 baseline.  
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
    num_workers = 2
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


train_loader = DataLoader(
    BirdDataset(train_df, train_mode=True),
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True,
)
valid_loader = DataLoader(
    BirdDataset(valid_df, train_mode=False),
    batch_size=CFG.batch_size * 2,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
)
"""
            ),
            md("## 4. Model And Training Loop"),
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
        ]
    )


def perch_notebook() -> dict:
    return notebook(
        [
            md(
                """
# BirdCLEF+ 2026 - Google Perch v2 Probe

Purpose: extract frozen Perch embeddings and train a shallow PyTorch probe.  
Artifacts are written to `/kaggle/working/artifacts/perch_v2`.
"""
            ),
            md("## 1. Setup"),
            code(
                COMMON_STYLE
                + """
import librosa
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

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
"""
            ),
            md("## 2. Load Metadata"),
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
"""
            ),
            md("## 4. Extract Embeddings"),
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
        outputs = infer(**{input_name: tensor})
    else:
        outputs = infer(tensor)
    value = next(iter(outputs.values()))
    return np.asarray(value).astype(np.float32)


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
train_idx, valid_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    random_state=CFG.seed,
    stratify=y,
)

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
        ]
    )


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_data_eda.ipynb": eda_notebook(),
        "02_effnet_b0_baseline.ipynb": effnet_notebook(),
        "03_perch_v2_probe.ipynb": perch_notebook(),
    }
    for name, nb in outputs.items():
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
