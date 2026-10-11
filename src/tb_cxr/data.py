"""Dataset discovery, labels, split construction, and image loading."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

FILENAME_RE = re.compile(r"^(CHNCXR|MCUCXR)_(\d{4})_([01])\.png$", re.IGNORECASE)
SOURCES = ("shenzhen", "montgomery")


@dataclass(frozen=True)
class XrayRecord:
    path: Path
    source: str
    record_id: str
    label: int


def parse_filename(path: str | Path) -> XrayRecord:
    """Parse NLM dataset labels encoded in image filenames (0 normal, 1 abnormal/TB-associated)."""
    path = Path(path)
    match = FILENAME_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unrecognized NLM CXR filename: {path.name}")
    prefix, number, label = match.groups()
    source = "shenzhen" if prefix.upper() == "CHNCXR" else "montgomery"
    return XrayRecord(path=path, source=source,
                      record_id=f"{source}-{number}", label=int(label))


def discover_records(root: str | Path) -> pd.DataFrame:
    """Discover PNG records below data/raw and validate names, uniqueness and readability."""
    root = Path(root)
    paths = sorted(p for p in root.rglob("*.png") if FILENAME_RE.fullmatch(p.name))
    if not paths:
        raise FileNotFoundError(
            f"No supported NLM images found below {root}. Run scripts/download_dataset.py first."
        )
    rows = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        record = parse_filename(path)
        key = (record.source, record.record_id)
        if key in seen:
            raise ValueError(f"Duplicate record ID: {key}")
        seen.add(key)
        # Lightweight format validation; full pixel decoding occurs in the Dataset.
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"Unreadable image: {path}") from exc
        rows.append({
            "path": path.as_posix(),
            "source": record.source,
            "record_id": record.record_id,
            "label": record.label,
        })
    return pd.DataFrame(rows).sort_values(["source", "record_id"]).reset_index(drop=True)


def make_stratified_splits(
    manifest: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.DataFrame:
    """Create source-specific, class-stratified splits; record IDs never cross partitions."""
    ratios = (train_fraction, validation_fraction, test_fraction)
    if any(v <= 0 for v in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train/validation/test fractions must be positive and sum to 1")
    required = {"path", "source", "record_id", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest requires columns {sorted(required)}")
    parts = []
    for source, frame in manifest.groupby("source", sort=True):
        frame = frame.reset_index(drop=True)
        if frame["label"].nunique() != 2 or frame["label"].value_counts().min() < 3:
            raise ValueError(f"{source} needs at least 3 records from each class for stratified splits")
        train, rest = train_test_split(
            frame, test_size=(validation_fraction + test_fraction),
            stratify=frame["label"], random_state=seed,
        )
        val_share = validation_fraction / (validation_fraction + test_fraction)
        val, test = train_test_split(
            rest, test_size=(1.0 - val_share), stratify=rest["label"], random_state=seed,
        )
        for split, block in (("train", train), ("validation", val), ("test", test)):
            out = block.copy()
            out["split"] = split
            parts.append(out)
    result = pd.concat(parts, ignore_index=True).sort_values(["source", "split", "record_id"]).reset_index(drop=True)
    # Fail closed if a case identifier accidentally appears more than once or crosses a split.
    if result.duplicated(["source", "record_id"]).any():
        raise ValueError("Duplicate case IDs in manifest")
    counts = result.groupby(["source", "record_id"])["split"].nunique()
    if (counts > 1).any():
        raise AssertionError("Record leakage across splits")
    return result


def save_manifest(root: str | Path, output: str | Path, seed: int = 42,
                  train_fraction: float = 0.70, validation_fraction: float = 0.15,
                  test_fraction: float = 0.15) -> pd.DataFrame:
    root, output = Path(root), Path(output)
    manifest = discover_records(root)
    # Store project-relative paths for portability when the folder is moved.
    base = root.parent.parent.resolve() if root.name == "raw" else root.resolve()
    manifest["path"] = [Path(p).resolve().relative_to(base).as_posix() for p in manifest["path"]]
    splits = make_stratified_splits(manifest, seed, train_fraction, validation_fraction, test_fraction)
    output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output, index=False)
    summary = splits.groupby(["source", "split", "label"]).size().rename("count").reset_index()
    summary.to_csv(output.with_name("split_summary.csv"), index=False)
    return splits


def image_transform(image_size: int, train: bool = False):
    """Keep radiograph aspect ratio, pad to square, and apply ImageNet normalization."""
    ops: list = []
    if train:
        # Conservative geometry-only augmentation; do not flip laterality.
        ops.append(transforms.RandomRotation(degrees=4, fill=0))
    ops.extend([
        transforms.Lambda(lambda im: ImageOps.pad(
            im.convert("RGB"), (image_size, image_size), method=Image.Resampling.BICUBIC,
            color=(0, 0, 0), centering=(0.5, 0.5))),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(ops)


class XrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: str | Path = ".", image_size: int = 224,
                 train: bool = False):
        self.frame = frame.reset_index(drop=True).copy()
        self.root = Path(root)
        self.transform = image_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        return self.transform(image), torch.tensor(int(row["label"]), dtype=torch.long), str(row["record_id"])


def source_counts(manifest: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {str(source): {str(int(label)): int(count) for label, count in frame["label"].value_counts().items()}
            for source, frame in manifest.groupby("source")}
