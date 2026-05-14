import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from project_paths import get_artifact_root
from sa_sam2 import SASAM2
from dataset import TestDataset


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                help="Path to the SA-SAM2 checkpoint.")
parser.add_argument("--test-images", "--test_image_path", dest="test_images", type=str, required=True,
                    help="Directory containing test images.")
parser.add_argument("--test-masks", "--test_gt_path", dest="test_masks", type=str, required=True,
                    help="Directory containing test masks.")
parser.add_argument("--save-dir", "--save_path", dest="save_dir", type=str,
                    default=str(get_artifact_root() / "predictions"),
                    help="Directory used to save predicted masks.")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_images, args.test_masks, 352)
checkpoint = torch.load(args.checkpoint, map_location=device)
model = SASAM2().to(device)
config = checkpoint.get("config", {})
if config:
    model.set_structural_prior_cfg(**config)
model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
model.eval()
if device.type == "cuda":
    model.cuda()
os.makedirs(args.save_dir, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        res, _, _ = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print("Saving " + name)
        imageio.imsave(os.path.join(args.save_dir, name[:-4] + ".png"), res)
