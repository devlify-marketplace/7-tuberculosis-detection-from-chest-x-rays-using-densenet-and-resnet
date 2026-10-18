# Tuberculosis Detection from Chest X-Rays Using DenseNet and ResNet

**Topic 7** from the supplied final-year project list. This is a complete, runnable research prototype that compares ImageNet-initialized **DenseNet121** and **ResNet50** for the NLM chest-radiograph task and measures transfer between the Shenzhen and Montgomery County collections. It includes official-source data acquisition, reproducible splits, training/evaluation, metrics with uncertainty intervals, a local inference API and upload demo, tests, project report, model card and data provenance notes.

> **Research-only scope.** Labels distinguish normal images from images marked abnormal/TB-associated in the collection. This is not a validated TB diagnosis, medical device, or replacement for radiologist assessment, laboratory testing or clinical judgement. Do not use its outputs for patient care.

## Devlify

An academic project in the [Devlify GitHub collection](https://github.com/devlify-marketplace).

[Visit Devlify](https://devlify.com.ng) · [Browse the marketplace](https://devlify.com.ng/marketplace)

## What is included

- `scripts/download_dataset.py` — retrieves the official NLM PNG files directly, verifies each image, and writes a SHA-256 inventory. The multi-gigabyte source data are **not** included in this code bundle.
- `scripts/prepare_data.py` and `src/tb_cxr/data.py` — parse NLM filename labels, validate image files, and build deterministic per-source stratified train/validation/test partitions (70/15/15 by default).
- `src/tb_cxr/model.py` — matching two-class DenseNet121 and ResNet50 heads; ImageNet weights and frozen-backbone transfer learning are the defaults.
- `src/tb_cxr/experiment.py` — trains both architectures on each source in turn (four fits), evaluates on that source's untouched test split, then evaluates on **all** records from the other source without adaptation.
- `src/tb_cxr/train.py` and `metrics.py` — early stopping, retained best validation-loss checkpoint, predictions, confusion matrices, sensitivity, specificity, accuracy, balanced accuracy, precision, F1, AUROC, MCC, and bootstrap confidence intervals.
- `src/tb_cxr/api.py` + `web/index.html` — local CPU inference API and browser demo. Uploads are processed in memory and are not written to disk.
- `tests/` — offline checks for file-label parsing, split reproducibility and leakage, metrics, model output shapes, and API behavior.
- `docs/` — academic-style report, data provenance/reuse notes, model card, results, architecture diagram source/image and a PDF report.

## Setup

Python 3.10–3.12 is supported. Python packages and pretrained weights require internet access on first install/run. CPU is supported; CUDA is optional. If your package index does not provide a compatible CPU PyTorch wheel, install a matching `torch`/`torchvision` pair using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing this package.

```bash
cd /workspace/topic-07-tb-detection-densenet-resnet
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[test]'
```

## Download and prepare the original data

The National Library of Medicine source contains 662 Shenzhen and 138 Montgomery County frontal radiographs as described in the dataset paper. Image files are several megabytes each; allow multiple gigabytes of disk space and sufficient download time.

```bash
python scripts/download_dataset.py --source all --workers 6
python scripts/prepare_data.py --seed 42
```

`data/raw/` holds the downloaded images and checksum inventory; `data/processed/splits.csv` and `split_summary.csv` hold the portable manifest. Neither dataset images nor patient-level data are redistributed in the source ZIP. Review [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) and current NLM terms before reuse.

## Run the comparison

```bash
python -m tb_cxr.experiment --root . --epochs 8 --batch-size 8 --num-workers 2
```

By default, this fits four models: each of the two architectures trained once on each source. ImageNet weights are fetched automatically by Torchvision if they are not cached. For an offline run with randomly initialized weights (software smoke test only), add `--no-pretrained`; do not describe such a run as a pretrained benchmark. To update all backbone layers rather than train just the new classification head, use `--no-freeze-backbone` and document that choice.

The CLI reads `configs/default.json`; any supplied command-line option overrides that file. A previously generated `data/processed/splits.csv` is reused unchanged on later runs. To intentionally change the split seed or ratios, rerun `scripts/prepare_data.py` with the new values or remove the old manifest before preparing it again.

Results are written to:

- `artifacts/comparison_summary.csv` and `.json` — eight evaluation rows (two architectures × two training sources × internal/external evaluation).
- `artifacts/generalization_gaps.csv` and `artifacts/comparison_metrics.png` — paired external-minus-internal differences and a confidence-interval chart.
- `artifacts/runs/train_<source>/<architecture>/best_model.pt` — best validation-loss checkpoint.
- The same run directory contains model metadata, training history/plot, internal and external prediction CSVs, metric JSON and confusion-matrix plots.

The completed reference run on all 800 official images produced internal AUROC values of 0.870–0.907 and cross-source AUROC values of 0.687–0.739; external accuracy ranged from 0.558 to 0.653. The drop and threshold-dependent sensitivity/specificity trade-offs are detailed in [`docs/RESULTS.md`](docs/RESULTS.md). This is a research benchmark only, not clinical validation.

An internal test score measures performance only on the held-out source partition. An external score uses the entire second source, which was excluded from that fit. These collections are small and demographically/institutionally distinct; uncertainty intervals and per-source results matter more than a single pooled accuracy. Do not change the external set into training data for the reported transfer evaluation.

For a quick engineering smoke test, use a separate copied project folder, one architecture, a very small `--image-size`, and `--epochs 1`. Smoke-test numbers are not reportable research results.

## Run the local inference demo

After at least one training run:

```bash
uvicorn tb_cxr.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. API routes: `GET /api/health`, `GET /api/models`, and `POST /api/predict` (multipart field `file`, optional `model_id` query parameter). The service accepts images up to 8 MB. Scores are uncalibrated model outputs, not calibrated clinical probabilities. The API does not retain uploaded bytes or history.

## Tests

```bash
pytest -q
```

The tests do not download radiographs or pretrained model weights. Small synthetic fixtures are used only to unit-test filename parsing and split mechanics; they are never used in reported experiments. Other checks cover network-output shapes with random initialization, metric behavior, and the API's no-checkpoint response. Tests are engineering checks, not medical validation.

## Project documents

- [Final-year project report](docs/PROJECT_REPORT.md) · [PDF](docs/PROJECT_REPORT.pdf)
- [Results and interpretation](docs/RESULTS.md)
- [Data provenance and reuse notes](docs/DATA_PROVENANCE.md)
- [Model card](docs/MODEL_CARD.md)
- [Architecture diagram](docs/ARCHITECTURE.png) · [source](docs/ARCHITECTURE.mmd)
