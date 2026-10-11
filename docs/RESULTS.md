# Experimental results

## Experimental protocol and data split

The original NLM images were downloaded from the official public directories and verified as readable PNGs: 662 Shenzhen images and 138 Montgomery County images (800 total; SHA-256 inventory in `data/raw/SHA256SUMS.txt`). The deterministic seed-42, source-stratified split contains:

| Source | Train (normal / abnormal) | Validation (normal / abnormal) | Test (normal / abnormal) | Total |
|---|---:|---:|---:|---:|
| Shenzhen | 463 (228 / 235) | 99 (49 / 50) | 100 (49 / 51) | 662 |
| Montgomery County | 96 (56 / 40) | 21 (12 / 9) | 21 (12 / 9) | 138 |

Four fits were run on CPU with PyTorch 2.7.1 and Torchvision 0.22.1: ImageNet-default DenseNet121 and ResNet50, frozen feature backbones, 224 × 224 input, batch size 8, AdamW (learning rate 0.0005, weight decay 0.0001), up to 8 epochs, early stopping patience 3, and seed 42. Checkpoint selection used validation loss. All reported thresholded results use the fixed 0.5 threshold. Each external evaluation uses every image from the other source and did not contribute to that fit, its checkpoint selection, or the threshold.

Intervals are class-stratified percentile bootstrap 95% confidence intervals using 1,000 replicates and seed 42. They reflect finite-sample uncertainty; they do not correct dataset bias, label noise, shortcut learning, or unmeasured domain differences. Values in the table are fractions on the 0–1 scale.

## Main results

| Train → evaluation source | Architecture | N | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | AUROC (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| Montgomery → Montgomery (internal) | DenseNet121 | 21 | 0.857 (0.714–1.000) | 0.778 (0.444–1.000) | 0.917 (0.750–1.000) | 0.880 (0.657–1.000) |
| Montgomery → Shenzhen (external) | DenseNet121 | 662 | 0.628 (0.594–0.666) | 0.619 (0.568–0.670) | 0.638 (0.586–0.687) | 0.687 (0.649–0.729) |
| Montgomery → Montgomery (internal) | ResNet50 | 21 | 0.810 (0.667–0.952) | 0.556 (0.222–0.889) | 1.000 (1.000–1.000) | 0.907 (0.750–1.000) |
| Montgomery → Shenzhen (external) | ResNet50 | 662 | 0.653 (0.616–0.687) | 0.690 (0.640–0.741) | 0.613 (0.555–0.666) | 0.687 (0.648–0.724) |
| Shenzhen → Shenzhen (internal) | DenseNet121 | 100 | 0.830 (0.760–0.890) | 0.725 (0.607–0.843) | 0.939 (0.857–1.000) | 0.900 (0.834–0.950) |
| Shenzhen → Montgomery (external) | DenseNet121 | 138 | 0.558 (0.493–0.623) | 0.931 (0.862–0.983) | 0.288 (0.188–0.388) | 0.731 (0.640–0.803) |
| Shenzhen → Shenzhen (internal) | ResNet50 | 100 | 0.800 (0.730–0.870) | 0.686 (0.569–0.804) | 0.918 (0.837–0.980) | 0.870 (0.796–0.934) |
| Shenzhen → Montgomery (external) | ResNet50 | 138 | 0.580 (0.507–0.652) | 0.862 (0.776–0.948) | 0.375 (0.275–0.488) | 0.739 (0.656–0.823) |

## Confusion-matrix counts

| Train → evaluation source | Architecture | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|
| Montgomery → Montgomery (internal) | DenseNet121 | 11 | 1 | 2 | 7 |
| Montgomery → Shenzhen (external) | DenseNet121 | 208 | 118 | 128 | 208 |
| Montgomery → Montgomery (internal) | ResNet50 | 12 | 0 | 4 | 5 |
| Montgomery → Shenzhen (external) | ResNet50 | 200 | 126 | 104 | 232 |
| Shenzhen → Shenzhen (internal) | DenseNet121 | 46 | 3 | 14 | 37 |
| Shenzhen → Montgomery (external) | DenseNet121 | 23 | 57 | 4 | 54 |
| Shenzhen → Shenzhen (internal) | ResNet50 | 45 | 4 | 16 | 35 |
| Shenzhen → Montgomery (external) | ResNet50 | 30 | 50 | 8 | 50 |

![Internal and external performance with 95% stratified bootstrap intervals](../artifacts/comparison_metrics.png)

## Interpretation

- **Both architectures lost performance on the other source.** Internal AUROC was 0.870–0.907, versus 0.687–0.739 externally. Accuracy fell by 15.7–27.2 percentage points depending on training source and architecture. This supports a material cross-dataset generalisation gap in this benchmark.
- **No architecture dominates in every setting.** On Montgomery-trained transfer to Shenzhen, DenseNet121 and ResNet50 had almost identical AUROC (0.687 each); ResNet50 had higher accuracy (0.653 vs. 0.628). On Shenzhen-trained transfer to Montgomery, ResNet50 had slightly higher AUROC (0.739 vs. 0.731), but its confidence interval overlaps broadly with DenseNet121's. These are descriptive comparisons, not evidence of a statistically significant architecture winner.
- **At threshold 0.5, transfer to Montgomery was high-sensitivity but low-specificity.** Sensitivity was 0.931 for DenseNet121 and 0.862 for ResNet50, while specificity was only 0.288 and 0.375. The models over-called the positive, abnormal/TB-associated label in the external Montgomery set. That trade-off would have operational consequences in a real screening setting, but no clinical conclusions can be drawn from this prototype.
- **The internal Montgomery estimates are unstable.** The held-out set has only 21 images; its wide intervals (for example, DenseNet sensitivity 0.444–1.000) make point estimates particularly uncertain. ResNet50's internal specificity interval is degenerate because it made no false positives among the 12 normal cases in that small split.
- The target label is the dataset's normal/abnormal-with-TB-associated-reading convention, not an independent microbiological gold standard. Scores are uncalibrated, data are institutionally limited, and subgroup or prospective performance was not evaluated.

## Files for audit and reproduction

The exact per-image predictions, per-split metrics and confusion plots are in `artifacts/runs/train_<source>/<architecture>/`. The aggregate table with flattened interval bounds is `artifacts/comparison_summary.csv`; `artifacts/comparison_summary.json` preserves the nested metrics; `artifacts/generalization_gaps.csv` records external-minus-internal changes. The downloadable chart is `artifacts/comparison_metrics.png`. Recreate the experiment using the commands in the top-level README after downloading the source data.

> **Clinical boundary:** these results do not validate an automated TB diagnostic or triage tool. They are a small public-dataset research benchmark only.
