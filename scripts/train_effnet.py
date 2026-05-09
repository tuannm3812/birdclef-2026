from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from birdclef2026.config import ensure_dir, load_config
from birdclef2026.data import label_maps, read_metadata
from birdclef2026.dataset import AudioConfig, BirdDataset
from birdclef2026.models import BirdClassifier
from birdclef2026.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/effnet_b0.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    train_cfg = cfg["train"]
    fold = int(args.fold if args.fold is not None else cfg.get("folds", {}).get("fold", 0))
    data_root = Path(args.data_root or cfg["data_root"])
    output_dir = ensure_dir(args.output_dir or cfg["output_dir"])
    folds_path = Path(cfg.get("processed_dir", "data/processed")) / "train_folds.csv"

    if folds_path.exists():
        df = pd.read_csv(folds_path)
    else:
        df, _, _ = read_metadata(data_root)
        df["fold"] = 0

    labels = sorted(df["primary_label"].unique())
    label_to_idx, idx_to_label = label_maps(labels)
    (output_dir / "labels.json").write_text(json.dumps(idx_to_label, indent=2), encoding="utf-8")

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    if train_cfg.get("max_train_samples"):
        train_df = train_df.sample(int(train_cfg["max_train_samples"]), random_state=int(cfg.get("seed", 42)))
    if train_cfg.get("max_valid_samples"):
        valid_df = valid_df.sample(int(train_cfg["max_valid_samples"]), random_state=int(cfg.get("seed", 42)))

    audio_cfg = AudioConfig(**cfg["audio"])
    train_ds = BirdDataset(train_df, label_to_idx, audio_cfg, train=True)
    valid_ds = BirdDataset(valid_df, label_to_idx, audio_cfg, train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = make_loader(train_ds, int(train_cfg["batch_size"]), True, num_workers, device)
    valid_loader = make_loader(valid_ds, int(train_cfg["batch_size"]) * 2, False, num_workers, device)
    model = BirdClassifier(
        backbone=train_cfg.get("backbone", "efficientnet_b0"),
        num_classes=len(labels),
        pretrained=bool(train_cfg.get("pretrained", True)),
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    best_acc = 0.0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
        )
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "config": cfg,
                    "fold": fold,
                    "valid_acc": best_acc,
                },
                output_dir / "best_effnet_b0.pt",
            )

    print(f"Best valid accuracy: {best_acc:.4f}")


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for x, y in tqdm(loader, desc="valid", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += x.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


if __name__ == "__main__":
    main()
