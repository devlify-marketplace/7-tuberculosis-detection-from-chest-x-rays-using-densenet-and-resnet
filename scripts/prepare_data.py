#!/usr/bin/env python3
"""Validate downloaded NLM images and write train/validation/test manifests."""
from __future__ import annotations
import argparse
from pathlib import Path
from tb_cxr.data import save_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    root = args.root.resolve()
    out = save_manifest(root / "data/raw", root / "data/processed/splits.csv", args.seed,
                        args.train_fraction, args.validation_fraction, args.test_fraction)
    print(f"Saved {len(out)} records to {root / 'data/processed/splits.csv'}")
    print(out.groupby(["source", "split", "label"]).size().to_string())

if __name__ == "__main__":
    main()
