"""Run paired DenseNet121/ResNet50 training and cross-dataset evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from .data import make_stratified_splits, discover_records
from .model import SUPPORTED_ARCHITECTURES
from .train import fit_model, evaluate_checkpoint


def run(root: Path, *, seed=42, image_size=224, batch_size=8, epochs=8,
        learning_rate=5e-4, weight_decay=1e-4, patience=3, num_workers=2,
        train_fraction=0.70, validation_fraction=0.15, test_fraction=0.15,
        decision_threshold=0.5,
        pretrained=True, freeze_backbone=True, architectures=SUPPORTED_ARCHITECTURES,
        device=None):
    root = root.resolve(); data_dir = root / "data/raw"
    split_path = root / "data/processed/splits.csv"
    if split_path.exists():
        splits = pd.read_csv(split_path)
        # Revalidate that the stored split has no accidental per-record leakage.
        if splits.duplicated(["source", "record_id"]).any():
            raise ValueError("Stored manifest contains duplicate record IDs")
    else:
        records = discover_records(data_dir)
        base = root.resolve()
        records["path"] = [Path(p).resolve().relative_to(base).as_posix() for p in records.path]
        splits = make_stratified_splits(records, seed, train_fraction,
                                        validation_fraction, test_fraction)
        split_path.parent.mkdir(parents=True, exist_ok=True); splits.to_csv(split_path, index=False)
    sources = sorted(splits.source.unique().tolist())
    if sources != ["montgomery", "shenzhen"]:
        raise ValueError(f"Expected both sources (montgomery, shenzhen), found: {sources}")
    results = []
    for train_source in sources:
        source_frame = splits[splits.source == train_source]
        train_frame = source_frame[source_frame.split == "train"]
        val_frame = source_frame[source_frame.split == "validation"]
        internal = source_frame[source_frame.split == "test"]
        external_source = next(s for s in sources if s != train_source)
        external = splits[splits.source == external_source]
        for architecture in architectures:
            run_dir = root / "artifacts/runs" / f"train_{train_source}" / architecture
            print(f"\n=== {architecture}: train={train_source}, external={external_source} ===", flush=True)
            checkpoint, _ = fit_model(
                train_frame, val_frame, root, architecture, run_dir,
                seed=seed, image_size=image_size, batch_size=batch_size, epochs=epochs,
                learning_rate=learning_rate, weight_decay=weight_decay, patience=patience,
                num_workers=num_workers, pretrained=pretrained, freeze_backbone=freeze_backbone,
                device=device,
            )
            internal_metrics = evaluate_checkpoint(checkpoint, internal, root, run_dir,
                f"internal_{train_source}_test", decision_threshold, batch_size, num_workers, device)
            external_metrics = evaluate_checkpoint(checkpoint, external, root, run_dir,
                f"external_{external_source}", decision_threshold, batch_size, num_workers, device)
            for result, split_name in ((internal_metrics, "internal_test"), (external_metrics, "external")):
                result.update({"train_source": train_source, "evaluation_source":
                               train_source if split_name == "internal_test" else external_source,
                               "evaluation_type": split_name, "checkpoint": str(checkpoint.relative_to(root))})
                results.append(result)
            print(f"  internal: acc={internal_metrics['accuracy']:.3f} sens={internal_metrics['sensitivity']:.3f} "
                  f"spec={internal_metrics['specificity']:.3f} auc={internal_metrics['roc_auc']}", flush=True)
            print(f"  external: acc={external_metrics['accuracy']:.3f} sens={external_metrics['sensitivity']:.3f} "
                  f"spec={external_metrics['specificity']:.3f} auc={external_metrics['roc_auc']}", flush=True)
    summary = pd.DataFrame(results)
    for metric in ("accuracy", "sensitivity", "specificity", "roc_auc"):
        summary[f"{metric}_ci95_low"] = summary["confidence_intervals_95"].map(
            lambda interval: interval.get(metric, [None, None])[0] if interval and interval.get(metric) else None)
        summary[f"{metric}_ci95_high"] = summary["confidence_intervals_95"].map(
            lambda interval: interval.get(metric, [None, None])[1] if interval and interval.get(metric) else None)
    summary.to_csv(root / "artifacts/comparison_summary.csv", index=False)
    (root / "artifacts/comparison_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    gaps = []
    for (train_source, architecture), frame in summary.groupby(["train_source", "architecture"]):
        internal = frame[frame["evaluation_type"] == "internal_test"].iloc[0]
        external = frame[frame["evaluation_type"] == "external"].iloc[0]
        row = {"train_source": train_source, "architecture": architecture,
               "internal_source": internal["evaluation_source"],
               "external_source": external["evaluation_source"]}
        for metric in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc"):
            row[f"internal_{metric}"] = internal[metric]
            row[f"external_{metric}"] = external[metric]
            row[f"external_minus_internal_{metric}"] = (
                float(external[metric] - internal[metric])
                if pd.notna(external[metric]) and pd.notna(internal[metric]) else None)
        gaps.append(row)
    pd.DataFrame(gaps).to_csv(root / "artifacts/generalization_gaps.csv", index=False)

    group_keys = [(source, arch) for source in sources for arch in architectures]
    labels = [f"{source.title()}\n{arch}" for source, arch in group_keys]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, metric in zip(axes.flat, ("accuracy", "sensitivity", "specificity", "roc_auc")):
        positions = np.arange(len(group_keys)); width = 0.36
        for offset, evaluation_type, color, label in (
            (-width / 2, "internal_test", "#2374ab", "Internal test"),
            (width / 2, "external", "#e07a2d", "External source"),
        ):
            subset = summary[summary.evaluation_type == evaluation_type].set_index(["train_source", "architecture"])
            vals, lows, highs = [], [], []
            for key in group_keys:
                item = subset.loc[key]
                value, low, high = item[metric], item[f"{metric}_ci95_low"], item[f"{metric}_ci95_high"]
                vals.append(float(value) if pd.notna(value) else 0.0)
                lows.append(float(low) if pd.notna(low) else (float(value) if pd.notna(value) else 0.0))
                highs.append(float(high) if pd.notna(high) else (float(value) if pd.notna(value) else 0.0))
            values = np.asarray(vals); err = np.vstack([np.maximum(0, values - lows), np.maximum(0, highs - values)])
            ax.bar(positions + offset, values, width, label=label, color=color,
                   yerr=err, capsize=2, error_kw={"elinewidth": 0.8})
        ax.set_title(metric.replace("_", " ").title()); ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score"); ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend(frameon=False, fontsize=9)
    for ax in axes[-1]: ax.set_xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    fig.suptitle("Internal and cross-dataset performance (95% stratified bootstrap CI)")
    fig.tight_layout(); fig.savefig(root / "artifacts/comparison_metrics.png", dpi=170); plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON config; defaults to <root>/configs/default.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--architectures", nargs="+", choices=SUPPORTED_ARCHITECTURES,
                        default=None)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda; auto-detected by default")
    args = parser.parse_args()
    config_path = args.config or (args.root / "configs/default.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    def configured(attribute, key, default):
        value = getattr(args, attribute)
        return value if value is not None else config.get(key, default)
    settings = {
        "seed": configured("seed", "seed", 42),
        "image_size": configured("image_size", "image_size", 224),
        "batch_size": configured("batch_size", "batch_size", 8),
        "epochs": configured("epochs", "epochs", 8),
        "learning_rate": configured("learning_rate", "learning_rate", 5e-4),
        "weight_decay": configured("weight_decay", "weight_decay", 1e-4),
        "patience": configured("patience", "patience", 3),
        "num_workers": configured("num_workers", "num_workers", 2),
        "train_fraction": configured("train_fraction", "train_fraction", 0.70),
        "validation_fraction": configured("validation_fraction", "validation_fraction", 0.15),
        "test_fraction": configured("test_fraction", "test_fraction", 0.15),
        "decision_threshold": configured("decision_threshold", "decision_threshold", 0.5),
        "pretrained": configured("pretrained", "pretrained", True),
        "freeze_backbone": configured("freeze_backbone", "freeze_backbone", True),
        "architectures": args.architectures or config.get("architectures", list(SUPPORTED_ARCHITECTURES)),
        "device": args.device,
    }
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["image_size"] < 32:
        parser.error("epochs/batch-size must be positive and image-size at least 32")
    if abs(settings["train_fraction"] + settings["validation_fraction"] + settings["test_fraction"] - 1.0) > 1e-8:
        parser.error("train/validation/test fractions must sum to 1")
    if not 0 <= settings["decision_threshold"] <= 1:
        parser.error("decision threshold must be in [0, 1]")
    summary = run(args.root, **settings)
    print("\nSaved artifacts/comparison_summary.csv")
    print(summary[["train_source", "architecture", "evaluation_type", "evaluation_source",
                   "accuracy", "sensitivity", "specificity", "roc_auc"]].to_string(index=False))

if __name__ == "__main__":
    main()
