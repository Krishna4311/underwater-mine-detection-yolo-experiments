# Underwater Mine Detection — Side-Scan Sonar (YOLO11)

Fine-tuning YOLO11 for single-class mine detection on side-scan sonar imagery, with a controlled comparison of pretrained-backbone sources (COCO vs. sonar-domain) at different model sizes on a small (437-instance) real dataset.

## Why this project

Built to answer a specific question: on a genuinely small, real dataset, does model size or pretraining source matter more for transfer learning performance? Three training runs isolate these variables independently rather than changing them together. 
```
Note: This is a learning project, not an exhaustive study. Three runs were enough to isolate the two variables that mattered 
most here (model size, pretraining source) — more runs (different architectures, more epochs, larger sonar-pretrained datasets)
could extend this further, but weren't necessary to answer the specific question this project set out to ask. Depth on a few
controlled comparisons mattered more here than breadth across many. 
```

## Dataset

**SIMD Side-scan sonar Imaging for Mine Detection**
1,170 real side-scan sonar images (2010–2021), originally two classes
(MILCO / NOMBO). Not redistributed in this repo — download from the
source and prepare it locally using the script below.

- Source: [Figshare, DOI 10.6084/m9.figshare.24574879](https://doi.org/10.6084/m9.figshare.24574879)
- Cite the original authors if you use this dataset — see the Figshare page for their citation format.

**Sonar-pretrained backbone (Run 2)**
[Samyukta31/sonar_yolo](https://huggingface.co/Samyukta31/sonar_yolo)
— community-published YOLO11m weights pretrained on side-scan sonar imagery (SONARINTEL project). Unverified/non-peer-reviewed; used here for transfer-learning comparison only.

## Setup

```bash
pip install ultralytics
```

1. Download SIMD from the link above and extract it.
2. Run `prepare_dataset.py` from inside the extracted folder (the one containing `2010/`, `2015/`, `2017/`, `2018/`, `2021/`). This merges all years, collapses labels to a single `mine` class, and produces a stratified 80/20 train/val split.
3. Update `data.yaml`'s `path` field to point at the resulting `merged_dataset/` folder.
4. Run the [training notebooks](https://github.com/Krishna4311/underwater-mine-detection-yolo-experiments/blob/main/umd-yolo.ipynb).

## Contents

| File | Description |
|---|---|
| `prepare_dataset.py` | Merges SIMD's year-folders, collapses to single-class labels, stratified train/val split |
| `data.yaml` | YOLO dataset config (single class: `mine`) |
| `01_data_preparation.ipynb` | Dataset verification, ground-truth visualization |
| `02_training_experiments.ipynb` | Three training runs (see Results) |
| `03_evaluation_and_comparison.ipynb` | Metrics comparison, confusion matrices, prediction visualization |

## Results

| | Run 1: COCO, YOLO11n | Run 2: Sonar-pretrained, YOLO11m | Run 3: COCO, YOLO11m |
|---|---|---|---|
| Precision | 0.703 | 0.724 | 0.465 |
| Recall | 0.724 | 0.622 | 0.382 |
| mAP50 | 0.721 | 0.681 | 0.396 |
| mAP50-95 | 0.357 | 0.333 | 0.174 |

At this dataset size, model size affected results more than pretraining source but sonar-domain pretraining showed a clear benefit once model size was held constant (Run 2 vs Run 3). 

Full writeup: [Medium Blog](https://medium.com/@mtarunp/i-tried-to-detect-underwater-mines-with-yolo-heres-what-a-small-dataset-can-and-can-t-do-f84cd75311ce?sharedUserId=mtarunp).

