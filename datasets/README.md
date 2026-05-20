# Datasets

## MoSL — Moroccan Sign Language Video Dataset

**Source:** Ben Zaid et al. (2026). Mendeley Data.  
**DOI:** [10.17632/23phgyt3mt.1](https://doi.org/10.17632/23phgyt3mt.1)

| Stat | Value |
|------|-------|
| Total clips | 2,216 MP4 files |
| Unique signs | 1,631 |
| Resolution | 460 × 460 px |
| Train / Val / Test | 1,674 / 430 / 112 |
| Singletons (1 clip/sign) | 74% |

### Download

```bash
# Download from Mendeley and place under:
data/vedios-dataset/<category>/<clip>.mp4
```

The raw video files are excluded from this repository (221 MB).  
Processed metadata is included in `data/labels.csv` and `data/splits.csv`.

## Processed Keypoints

Extracted by the OpenPose pipeline (requires Docker + GPU):

```bash
docker/run.sh python scripts/extract_dataset.py
```

Output: `data/processed/keypoints_2d/<category>/<clip>.npz`
