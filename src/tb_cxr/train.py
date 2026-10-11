"""Training and evaluation helpers for reproducible transfer-learning experiments."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from .data import XrayDataset
from .metrics import calculate_metrics
from .model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(os.environ.get("TB_CXR_TORCH_THREADS", "4"))))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _loader(frame, root, image_size, batch_size, train, workers):
    return DataLoader(
        XrayDataset(frame, root=root, image_size=image_size, train=train),
        batch_size=batch_size, shuffle=train, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )


def _loss_epoch(model, loader, criterion, device, optimizer=None, freeze_backbone=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        # Keep frozen ImageNet batch-normalization statistics fixed during head-only fitting.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0; count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images); loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(labels); count += len(labels)
    return total_loss / max(count, 1)


def fit_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    root: str | Path,
    architecture: str,
    output_dir: str | Path,
    *,
    seed: int = 42,
    image_size: int = 224,
    batch_size: int = 8,
    epochs: int = 8,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 3,
    num_workers: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(architecture, pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)
    train_loader = _loader(train_frame, root, image_size, batch_size, True, num_workers)
    val_loader = _loader(validation_frame, root, image_size, batch_size, False, num_workers)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_loss, stale, best_epoch = float("inf"), 0, 0
    checkpoint_path = output_dir / "best_model.pt"
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = _loss_epoch(model, train_loader, criterion, device, optimizer, freeze_backbone)
        val_loss = _loss_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"  epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({
                "architecture": architecture,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "image_size": image_size,
                "class_names": ["normal", "abnormal_tb_associated"],
                "positive_class": 1,
                "best_epoch": best_epoch,
                "validation_loss": best_loss,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    if history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history_frame.epoch, history_frame.train_loss, marker="o", label="Train")
        ax.plot(history_frame.epoch, history_frame.validation_loss, marker="o", label="Validation")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title=f"{architecture}: learning curve")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(output_dir / "training_history.png", dpi=150); plt.close(fig)
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"architecture": architecture, "pretrained_imagenet": pretrained,
                   "freeze_backbone": freeze_backbone, "image_size": image_size,
                   "best_epoch": best_epoch, "best_validation_loss": best_loss,
                   "device": str(device), "seed": seed,
                   "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
                   "training_config": {"batch_size": batch_size, "epochs_requested": epochs,
                       "learning_rate": learning_rate, "weight_decay": weight_decay,
                       "patience": patience, "num_workers": num_workers},
                   "training_records": int(len(train_frame)),
                   "validation_records": int(len(validation_frame)),
                   "positive_class": "abnormal_tb_associated"}, f, indent=2)
    return checkpoint_path, history_frame


def load_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(checkpoint["architecture"], pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def predict_frame(model, frame: pd.DataFrame, root: str | Path, image_size: int,
                  batch_size: int = 16, num_workers: int = 0, device: str | None = None):
    device = torch.device(device or next(model.parameters()).device)
    loader = _loader(frame, root, image_size, batch_size, False, num_workers)
    labels, probabilities, ids = [], [], []
    model.eval()
    with torch.no_grad():
        for images, y, record_ids in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels.extend(y.numpy().tolist()); probabilities.extend(probs.tolist()); ids.extend(record_ids)
    return np.asarray(labels), np.asarray(probabilities), ids


def evaluate_checkpoint(checkpoint_path: str | Path, frame: pd.DataFrame, root: str | Path,
                        output_dir: str | Path, split_name: str,
                        threshold: float = 0.5, batch_size: int = 16, num_workers: int = 0,
                        device: str | None = None) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_checkpoint(checkpoint_path, device or ("cuda" if torch.cuda.is_available() else "cpu"))
    y, p, ids = predict_frame(model, frame, root, checkpoint["image_size"], batch_size, num_workers, device)
    metrics = calculate_metrics(y, p, threshold)
    metrics.update({"split": split_name, "architecture": checkpoint["architecture"],
                    "n_records": len(frame), "positive_label": "abnormal_tb_associated"})
    predictions = pd.DataFrame({"record_id": ids, "y_true": y, "probability_abnormal": p,
                                "y_pred": (p >= threshold).astype(int)})
    predictions.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)
    with (output_dir / f"metrics_{split_name}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix): ax.text(j, i, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], xticklabels=["Normal", "Abnormal"], yticks=[0, 1],
           yticklabels=["Normal", "Abnormal"], xlabel="Predicted", ylabel="Reference",
           title=f"{checkpoint['architecture']} — {split_name}")
    fig.tight_layout(); fig.savefig(output_dir / f"confusion_{split_name}.png", dpi=150); plt.close(fig)
    return metrics
