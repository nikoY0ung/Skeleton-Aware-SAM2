# PAPER_CODE_ALIGNMENT

This note summarizes how the current repository implementation maps to the paper-facing method description.

## Implemented Paper-Critical Elements

- backbone:
  - frozen `SAM2 Hiera` image encoder with lightweight adapter tuning
- decoder:
  - U-Net-style multi-scale decoder with side outputs
- structural prior:
  - label-free image-derived vessel prior built from grayscale thresholding, morphology cleanup, skeletonization, and distance-based Gaussian spreading
- modulation:
  - multi-stage `GatedSkeletonSPADE` injection at coarse and subsequent decoder stages
- auxiliary regularization:
  - training-only auxiliary skeleton head with Dice-style skeleton supervision
- evaluation:
  - overlap metrics and skeleton-aware `clDice_skeleton`

## Important Public-Facing Renames

- old model file:
  - `SAM2UNet.py`
- new model file:
  - `sa_sam2.py`
- old training / inference / evaluation entry points:
  - `train.py`
  - `test.py`
  - `eval.py`
- new entry points:
  - `train_sa_sam2.py`
  - `infer_sa_sam2.py`
  - `evaluate_predictions.py`

## Remaining Simplifications vs Full Manuscript Narrative

- the current repository is focused on the vessel-segmentation method itself
- dataset protocol, manuscript figures, and paper table-generation scripts are not bundled here
- checkpoint selection is still train-loss based because no explicit validation split helper is built into this minimal public repo
- the implementation exposes configurable switches so baseline-like ablations can still be run from the same codebase

## Practical Interpretation

This repository is now suitable as the public project codebase for the paper method, with the core structural-prior mechanism and naming aligned to the manuscript story.
