# Skeleton-Aware SAM2 for Topology-Preserving Angiographic Vessel Segmentation

This repository contains the project code for **Skeleton-Aware SAM2 (SA-SAM2)**, a morphology-guided adaptation framework for coronary angiography vessel segmentation.

The method uses:

- a frozen `SAM2 Hiera` image encoder with lightweight adapters
- a U-Net-style decoder with multi-scale feature fusion
- a **label-free distance-aware skeleton prior** extracted from the input image
- **multi-stage Gated SkeletonSPADE (GS-SPADE)** modules that inject structural cues into decoder features
- a training-only auxiliary skeleton head for topology-sensitive regularization

## Repository Layout

- [sa_sam2.py](sa_sam2.py): main model definition
- [structural_priors.py](structural_priors.py): skeleton prior extraction, GS-SPADE, and skeleton losses
- [train_sa_sam2.py](train_sa_sam2.py): training entry point
- [infer_sa_sam2.py](infer_sa_sam2.py): inference script
- [evaluate_predictions.py](evaluate_predictions.py): quantitative evaluation
- [dataset.py](dataset.py): dataset loaders
- `sam2/`, `sam2_configs/`: vendored SAM2 backbone code and configs

## Requirements

Create an environment and install dependencies:

```bash
conda create -n sa-sam2 python=3.10
conda activate sa-sam2
pip install -r requirements.txt
```

## Data Layout

Set `SA_SAM2_DATA_ROOT` if your datasets live outside the repository. The training and evaluation scripts assume layouts such as:

```text
data/
├── TJ_train/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       ├── images/
│       └── masks/
├── TJ_test/
│   ├── images/
│   └── masks/
├── XCAD_test/
│   ├── images/
│   └── masks/
└── XCAV_test/
    ├── img/
    └── gt/
```

## Artifact Layout

To keep the repository clean for GitHub, checkpoints, predictions, metrics, and plots are written outside the repo by default:

```text
<repo>_artifacts/
├── checkpoints/
├── predictions/
├── metrics/
└── visualizations/
```

Override this with `SA_SAM2_ARTIFACTS_DIR=/path/to/artifacts`.

## Training

Download a compatible `SAM2` Hiera checkpoint from the official Segment Anything 2 repository, then run:

```bash
python train_sa_sam2.py \
  --hiera-checkpoint /path/to/sam2_hiera_large.pt \
  --train-images data/TJ_train/train/images/ \
  --train-masks data/TJ_train/train/masks/ \
  --save-dir ../SAM2-UNet-main_artifacts/checkpoints/sa_sam2 \
  --epochs 20 \
  --batch-size 12 \
  --use-gs-spade \
  --use-aux-skeleton-head \
  --skeleton-aux-loss-weight 0.15 \
  --prior-sigma 3.0
```

## Inference

```bash
python infer_sa_sam2.py \
  --checkpoint ../SAM2-UNet-main_artifacts/checkpoints/sa_sam2/best_model.pt \
  --test-images data/TJ_test/images/ \
  --test-masks data/TJ_test/masks/ \
  --save-dir ../SAM2-UNet-main_artifacts/predictions/TJ_test
```

## Evaluation

```bash
python evaluate_predictions.py \
  --dataset-name TJ_test \
  --pred-dir ../SAM2-UNet-main_artifacts/predictions/TJ_test \
  --gt-dir data/TJ_test/masks/
```

This script reports:

- Dice
- IoU
- Precision
- Recall
- Specificity
- Accuracy
- `clDice_skeleton`

## Notes

- The current implementation follows the paper story: a distance-aware image-derived skeleton prior, multi-stage GS-SPADE conditioning, and an auxiliary skeleton supervision branch.
- The auxiliary skeleton branch is used only during training and adds no inference-time cost.
- The repository intentionally excludes datasets and large experiment outputs from version control.