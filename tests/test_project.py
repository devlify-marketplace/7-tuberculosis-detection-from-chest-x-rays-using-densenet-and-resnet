from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from tb_cxr.data import make_stratified_splits, parse_filename
from tb_cxr.metrics import calculate_metrics
from tb_cxr.model import build_model
import tb_cxr.api as api_module


def test_parse_filename_maps_source_and_suffix():
    r = parse_filename("CHNCXR_0123_1.png")
    assert (r.source, r.record_id, r.label) == ("shenzhen", "shenzhen-0123", 1)
    r = parse_filename("MCUCXR_0042_0.png")
    assert (r.source, r.record_id, r.label) == ("montgomery", "montgomery-0042", 0)
    with pytest.raises(ValueError):
        parse_filename("unrelated.png")


def test_splits_are_stratified_reproducible_and_disjoint():
    rows = []
    for source, prefix, n in (("shenzhen", "CHNCXR", 80), ("montgomery", "MCUCXR", 40)):
        for i in range(n):
            label = i % 2
            rows.append({"path": f"{prefix}_{i:04d}_{label}.png", "source": source,
                         "record_id": f"{source}-{i:04d}", "label": label})
    frame = pd.DataFrame(rows)
    first = make_stratified_splits(frame, seed=17)
    second = make_stratified_splits(frame, seed=17)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(frame)
    assert not first.duplicated(["source", "record_id"]).any()
    assert first.groupby(["source", "record_id"]).split.nunique().max() == 1
    for _, block in first.groupby("source"):
        assert set(block.split) == {"train", "validation", "test"}
        assert all(block.groupby("split").label.nunique() == 2)


def test_split_rejects_bad_ratios():
    frame = pd.DataFrame([{"path":"x", "source":"s", "record_id":"1", "label":0}])
    with pytest.raises(ValueError):
        make_stratified_splits(frame, train_fraction=.8, validation_fraction=.3, test_fraction=.1)


def test_metrics_include_screening_rates_and_ci():
    result = calculate_metrics([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], n_bootstrap=200)
    assert result["accuracy"] == pytest.approx(.5)
    assert result["sensitivity"] == pytest.approx(.5)
    assert result["specificity"] == pytest.approx(.5)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert "confidence_intervals_95" in result
    assert 0 <= result["roc_auc"] <= 1


@pytest.mark.parametrize("architecture", ["densenet121", "resnet50"])
def test_model_shapes_and_frozen_head(architecture):
    model = build_model(architecture, pretrained=False, freeze_backbone=True).eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 2)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all("classifier" in name or name.startswith("fc.") for name in trainable)


def test_api_without_checkpoint_reports_unavailable(monkeypatch):
    monkeypatch.setattr(api_module, "available_models", lambda: [])
    client = TestClient(api_module.app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").json()["clinical_use"] is False
    img = Image.new("RGB", (32, 32), "black")
    data = io.BytesIO(); img.save(data, format="PNG")
    response = client.post("/api/predict", files={"file": ("cxr.png", data.getvalue(), "image/png")})
    assert response.status_code == 503
