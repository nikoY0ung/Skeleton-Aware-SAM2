from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from structural_priors import cldice_from_masks


def compute_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    pred_bin = pred > 127
    gt_bin = gt > 127

    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    tn = float(np.logical_and(~pred_bin, ~gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())

    return {
        "Dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "IoU": float((tp + eps) / (tp + fp + fn + eps)),
        "Precision": float((tp + eps) / (tp + fp + eps)),
        "Recall": float((tp + eps) / (tp + fn + eps)),
        "Specificity": float((tn + eps) / (tn + fp + eps)),
        "Accuracy": float((tp + tn + eps) / (tp + tn + fp + fn + eps)),
        "clDice_skeleton": float(cldice_from_masks(pred_bin, gt_bin, eps=eps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SA-SAM2 prediction masks.")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", type=str, required=True)
    parser.add_argument("--pred-dir", "--pred_path", dest="pred_dir", type=str, required=True)
    parser.add_argument("--gt-dir", "--gt_path", dest="gt_dir", type=str, required=True)
    args = parser.parse_args()

    pred_root = Path(args.pred_dir)
    gt_root = Path(args.gt_dir)
    names = sorted([n for n in os.listdir(gt_root) if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])

    totals = {k: 0.0 for k in ("Dice", "IoU", "Precision", "Recall", "Specificity", "Accuracy", "clDice_skeleton")}
    for idx, name in enumerate(names):
        pred_path = pred_root / f"{Path(name).stem}.png"
        gt_path = gt_root / name
        print(f"[{idx}] Processing {name}...")

        pred = imageio.imread(pred_path)
        gt = imageio.imread(gt_path)
        if pred.ndim == 3:
            pred = pred[..., 0]
        if gt.ndim == 3:
            gt = gt[..., 0]

        stats = compute_metrics(pred, gt)
        for key, value in stats.items():
            totals[key] += value

    count = max(len(names), 1)
    means = {k: v / count for k, v in totals.items()}
    print(args.dataset_name)
    for key in ("Dice", "IoU", "Precision", "Recall", "Specificity", "Accuracy", "clDice_skeleton"):
        print(f"{key}: {means[key]:.4f}")


if __name__ == "__main__":
    main()
