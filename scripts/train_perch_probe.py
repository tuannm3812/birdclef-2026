from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from birdclef2026.config import ensure_dir, load_config
from birdclef2026.data import label_maps
from birdclef2026.models import PerchProbe
from birdclef2026.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/perch_probe.yaml")
    parser.add_argument("--embeddings")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    probe_cfg = cfg["probe"]
    output_dir = ensure_dir(args.output_dir or cfg["output_dir"])
    embeddings_path = Path(args.embeddings or cfg["perch"]["embeddings_path"])

    data = np.load(embeddings_path, allow_pickle=True)
    x = to_2d_embeddings(data["embeddings"].astype(np.float32))
    labels = [str(label) for label in data["labels"]]
    label_to_idx, _ = label_maps(labels)
    y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)

    train_idx, valid_idx = safe_train_valid_split(y, seed=int(cfg.get("seed", 42)))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=int(probe_cfg["batch_size"]),
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[valid_idx]), torch.from_numpy(y[valid_idx])),
        batch_size=int(probe_cfg["batch_size"]) * 2,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerchProbe(
        embedding_dim=x.shape[1],
        num_classes=len(label_to_idx),
        hidden_dim=int(probe_cfg["hidden_dim"]),
        dropout=float(probe_cfg["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(probe_cfg["lr"]))

    best_acc = 0.0
    for epoch in range(1, int(probe_cfg["epochs"]) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        valid_acc = validate(model, valid_loader, device)
        print(f"epoch={epoch} valid_acc={valid_acc:.4f}")
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(
                {"model": model.state_dict(), "label_to_idx": label_to_idx, "config": cfg},
                output_dir / "best_perch_probe.pt",
            )

    print(f"Best valid accuracy: {best_acc:.4f}")
    zip_path = output_dir.parent / f"{output_dir.name}_artifacts.zip"
    zip_artifacts(output_dir, zip_path)
    print(f"Zipped artifacts to {zip_path}")


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    seen = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        seen += xb.size(0)
    return correct / max(seen, 1)


def safe_train_valid_split(targets: np.ndarray, seed: int, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    counts = pd.Series(targets).value_counts()
    rare_classes = set(counts[counts < 2].index)
    all_idx = np.arange(len(targets))
    rare_idx = np.array([idx for idx in all_idx if targets[idx] in rare_classes], dtype=np.int64)
    common_idx = np.array([idx for idx in all_idx if targets[idx] not in rare_classes], dtype=np.int64)
    train_common, valid_idx = train_test_split(
        common_idx,
        test_size=test_size,
        random_state=seed,
        stratify=targets[common_idx],
    )
    train_idx = np.concatenate([train_common, rare_idx])
    return train_idx, valid_idx


def to_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 2:
        return embeddings
    if embeddings.ndim == 3:
        return embeddings.mean(axis=1)
    return embeddings.reshape(embeddings.shape[0], -1)


def zip_artifacts(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(source_dir.parent))


if __name__ == "__main__":
    main()
