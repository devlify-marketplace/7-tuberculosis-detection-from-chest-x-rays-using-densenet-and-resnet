# Model card: DenseNet121 / ResNet50 TB-associated CXR research classifiers

## Model details

- **Purpose:** academic comparison of two transfer-learning architectures on the NLM Shenzhen and Montgomery County chest X-ray collections.
- **Model types:** Torchvision DenseNet121 and ResNet50, ImageNet pretrained by default, with a new two-class linear head. Backbone frozen by default.
- **Task:** image-level classification of normal (0) versus abnormal/TB-associated source label (1).
- **Inputs:** frontal chest X-ray PNG/JPEG rendered as RGB, aspect-ratio-preserving square padding, ImageNet normalization, default 224 × 224.
- **Outputs:** two-class softmax-derived score and thresholded research label. The score is not calibrated as a clinical probability.
- **Training recipe:** see `docs/PROJECT_REPORT.md` and per-checkpoint `model_metadata.json`.

## Intended use

Research, reproducible coding exercises, and comparison of in-domain versus cross-source evaluation. Results should be stated as performance on the defined benchmark and label convention.

## Out-of-scope use

Diagnosis, rule-out, triage, treatment, isolation decisions, public-health surveillance decisions, clinical workflow integration, patient-level risk communication, or replacing radiologist review and confirmatory testing. No clinical safety or effectiveness claim is made.

## Training and evaluation data

The official NLM Shenzhen and Montgomery County public datasets, as described by Jaeger et al. (2014). The paper reports 662 Shenzhen cases (326 normal, 336 with TB manifestations) and 138 Montgomery cases (80 normal, 58 with TB manifestations). Separate 70/15/15 stratified record splits are used for each source. Each fit's external dataset is untouched until evaluation. See `docs/DATA_PROVENANCE.md` for links, hashes and label cautions.

## Metrics

Sensitivity, specificity, accuracy, balanced accuracy, precision, F1, AUROC, MCC, confusion matrix and class-stratified 95% bootstrap intervals at fixed threshold 0.5. In the completed run, internal AUROC ranged from 0.870 to 0.907 and external AUROC from 0.687 to 0.739; external accuracy ranged from 0.558 to 0.653. When trained on Shenzhen and evaluated on Montgomery, external specificity was 0.288–0.375 at the fixed threshold. These are benchmark-only measurements, not estimates of clinical performance. See `docs/RESULTS.md` and generated artifacts for all measured values. Small test sets—especially the 138-image Montgomery collection—yield substantial uncertainty.

## Known limitations and risks

- Source labels are not equivalent to microbiological confirmation or active-TB diagnosis.
- Two small institutional datasets cannot establish broad geographic, demographic or scanner generalisation.
- Domain differences and dataset shortcuts may dominate the visual features learned by the model.
- The benchmark has limited evidence on demographic subgroups, calibration, reader performance or prospective clinical outcomes.
- Record-level separation is enforced; the filenames do not provide a unified cross-source patient identity key.
- The default ImageNet initialization and fixed threshold do not imply clinical validity.

## Ethical considerations and safeguards

The UI/API repeat the research-only disclaimer. The local API does not store uploaded image bytes. Avoid identifiable or unauthorised medical images. External clinical deployment would require new data governance, institutional ethics review, representative prospective validation, clinical risk analysis, calibration, subgroup evaluation and appropriate regulatory review.
