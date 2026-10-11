# Data provenance and reuse notes

## Primary source

U.S. National Library of Medicine (NLM), Lister Hill National Center for Biomedical Communications, [public Tuberculosis Chest X-ray Datasets index](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/).

| Source | Official image directory | Published count | Published label counts |
|---|---|---:|---|
| Shenzhen Hospital | [NLM Shenzhen CXR PNG folder](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/CXR_png/) | 662 | 326 normal; 336 with TB manifestations |
| Montgomery County | [NLM Montgomery CXR PNG folder](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/) | 138 | 80 normal; 58 with TB manifestations |

These counts and collection descriptions are from Jaeger et al. (2014), the source paper linked below. The official NLM directory pages expose individual PNGs. The downloader fetches only the radiograph PNGs needed for this experiment; it does not fetch the manual masks, annotation files or clinical reading documents.

## Labels and identifiers

Filenames use the patterns `CHNCXR_####_0.png`, `CHNCXR_####_1.png`, `MCUCXR_####_0.png` and `MCUCXR_####_1.png`. The project maps suffix 0 to `normal` (label 0) and suffix 1 to `abnormal_tb_associated` (label 1). This mapping captures the source's binary naming convention. It must not be restated as microbiologically confirmed active TB or a definitive patient-level diagnosis. The code uses the source prefix and record number as a unique record key and rejects duplicate keys. It cannot link patient identity across the two sources.

## Acquisition and integrity

Run `python scripts/download_dataset.py --source all --workers 6`. The script reads the current official HTML directory listings, downloads PNG files into `data/raw/<source>/CXR_png/`, verifies each file can be decoded as an image, and writes SHA-256 hashes and byte lengths to `data/raw/SHA256SUMS.txt`. Existing readable files are reused. The downloads are not included in the source ZIP due to their multi-gigabyte size and separate dataset reuse conditions.

After download, run `python scripts/prepare_data.py --seed 42`. The generated manifest uses project-relative image paths and deterministic source-stratified 70/15/15 record partitions. No image is duplicated across train, validation and test. The full source from which an external evaluation set comes is untouched by model fitting and checkpoint selection.

## Reuse and privacy

The source article reports that data providers de-identified the collections and that public release was exempted from IRB review for the described release. This does not remove all privacy, ethical, legal, or reuse obligations. Confirm the current NLM page's terms and your institution/supervisor's requirements before reuse or redistribution. Do not re-identify subjects. Do not upload identifiable medical images to the web demo; the local example processes uploaded bytes in memory and does not persist them.

## Citation

Jaeger S, Candemir S, Antani S, Wang Y-XJ, Lu P-X, Thoma G. *Two public chest X-ray datasets for computer-aided screening of pulmonary diseases.* Quantitative Imaging in Medicine and Surgery. 2014;4(6):475–477. [doi:10.3978/j.issn.2223-4292.2014.11.20](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4256233/).
