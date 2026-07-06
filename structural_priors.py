from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize


class GatedSkeletonSPADE(nn.Module):
    """SPADE-style feature modulation with a learnable spatial gate."""

    def __init__(
        self,
        norm_nc: int,
        label_nc: int = 1,
        nhidden: int = 64,
        num_groups: int = 8,
        gate_hidden: int = 32,
    ):
        super().__init__()
        if norm_nc % num_groups != 0:
            for g in range(num_groups, 0, -1):
                if norm_nc % g == 0:
                    num_groups = g
                    break

        self.param_free_norm = nn.GroupNorm(num_groups, norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.gate_net = nn.Sequential(
            nn.Conv2d(norm_nc + label_nc, gate_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=3, padding=1),
        )

        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.zeros_(self.mlp_beta.bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -2.0)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        if prior.shape[-2:] != x.shape[-2:]:
            prior = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)

        normalized = self.param_free_norm(x)
        act = self.mlp_shared(prior)
        gamma = self.mlp_gamma(act)
        beta = self.mlp_beta(act)

        gate_input = torch.cat((normalized, prior), dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return normalized * (1 + gate * gamma) + gate * beta


class SkeletonAuxHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _apply_morphology(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    opened = ndimage.binary_opening(mask, structure=structure)
    closed = ndimage.binary_closing(opened, structure=structure)
    return closed


def _distance_aware_prior_from_skeleton(skeleton: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    skeleton = skeleton.astype(np.float32)
    if skeleton.max() <= 0.0:
        return skeleton
    inv_skeleton = (skeleton < 0.5).astype(np.uint8)
    dist = ndimage.distance_transform_edt(inv_skeleton).astype(np.float32)
    dist[skeleton >= 0.5] = 0.0
    return np.exp(-(dist ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def build_skeleton_tensor_from_binary_mask(
    masks: torch.Tensor,
    clean: bool = True,
    kernel_size: int = 3,
) -> torch.Tensor:
    masks_np = masks.detach().float().cpu().numpy()
    out = []
    for mask in masks_np:
        vessel_mask = mask[0] > 0.5
        if clean:
            vessel_mask = _apply_morphology(vessel_mask, kernel_size=kernel_size)
        out.append(skeletonize(vessel_mask).astype(np.float32))
    return torch.from_numpy(np.stack(out, axis=0)[:, None, ...]).float()


def build_distance_aware_skeleton_prior_from_image(
    images: torch.Tensor,
    *,
    bright_quantile: float = 0.7,
    clean: bool = True,
    kernel_size: int = 3,
    sigma: float = 3.0,
) -> torch.Tensor:
    images_np = images.detach().float().cpu().numpy()
    priors = []
    for image in images_np:
        if image.shape[0] >= 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image[0]
        if gray.max() > 1.5:
            gray = gray / 255.0

        flat = gray.reshape(-1)
        threshold = np.median(flat) if bright_quantile <= 0.0 or bright_quantile >= 1.0 else float(np.quantile(flat, bright_quantile))
        vessel_mask = gray >= threshold
        if clean:
            vessel_mask = _apply_morphology(vessel_mask, kernel_size=kernel_size)

        skeleton = skeletonize(vessel_mask).astype(np.float32)
        priors.append(_distance_aware_prior_from_skeleton(skeleton, sigma=sigma))

    return torch.from_numpy(np.stack(priors, axis=0)[:, None, ...]).float()


def skeleton_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    inter = (pred * targets).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def cldice_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    pred_bin = pred_mask.astype(bool)
    gt_bin = gt_mask.astype(bool)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0

    pred_skel = skeletonize(pred_bin).astype(np.float32)
    gt_skel = skeletonize(gt_bin).astype(np.float32)
    topology_precision = ((pred_bin.astype(np.float32) * gt_skel).sum() + eps) / (gt_skel.sum() + eps)
    topology_recall = ((gt_bin.astype(np.float32) * pred_skel).sum() + eps) / (pred_skel.sum() + eps)
    return float((2.0 * topology_precision * topology_recall) / (topology_precision + topology_recall + eps))
