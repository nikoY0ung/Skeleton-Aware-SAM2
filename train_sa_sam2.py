import os
import argparse
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from project_paths import get_artifact_root
from sa_sam2 import SASAM2
from structural_priors import build_skeleton_tensor_from_binary_mask, skeleton_dice_loss


parser = argparse.ArgumentParser("Skeleton-Aware SAM2")
parser.add_argument("--hiera-checkpoint", "--hiera_path", dest="hiera_checkpoint", type=str, required=True,
                    help="Path to the SAM2 Hiera checkpoint.")
parser.add_argument("--train-images", "--train_image_path", dest="train_images", type=str, required=True,
                    help="Directory containing training images.")
parser.add_argument("--train-masks", "--train_mask_path", dest="train_masks", type=str, required=True,
                    help="Directory containing training masks.")
parser.add_argument("--save-dir", "--save_path", dest="save_dir", type=str,
                    default=str(get_artifact_root() / "checkpoints" / "sa_sam2"),
                    help="Directory used to store model checkpoints.")
parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=20,
                    help="Training epochs.")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)

parser.add_argument("--use-gs-spade", "--use_skeleton_spade", dest="use_gs_spade", action="store_true")
parser.add_argument("--use-aux-skeleton-head", "--use_skeleton_aux_head", dest="use_aux_skeleton_head", action="store_true")
parser.add_argument("--skeleton_aux_loss_weight", type=float, default=0.15)
parser.add_argument("--prior-dark-quantile", "--skeleton_dark_quantile", dest="prior_dark_quantile", type=float, default=0.25)
parser.add_argument("--prior-kernel", "--skeleton_kernel", dest="prior_kernel", type=int, default=3)
parser.add_argument("--prior-sigma", type=float, default=3.0)
parser.add_argument("--no-prior-clean", "--no_skeleton_clean", dest="no_prior_clean", action="store_true")
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def main(args):    
    dataset = FullDataset(args.train_images, args.train_masks, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SASAM2(args.hiera_checkpoint)
    model.set_structural_prior_cfg(
        use_gs_spade=bool(args.use_gs_spade),
        use_aux_skeleton_head=bool(args.use_aux_skeleton_head),
        prior_dark_quantile=float(args.prior_dark_quantile),
        prior_kernel=int(args.prior_kernel),
        prior_clean=not bool(args.no_prior_clean),
        prior_sigma=float(args.prior_sigma),
        skeleton_aux_loss_weight=float(args.skeleton_aux_loss_weight),
    )
    model.to(device)
    optim = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epochs, eta_min=1.0e-7)
    os.makedirs(args.save_dir, exist_ok=True)
    run_config = {
        "use_gs_spade": bool(args.use_gs_spade),
        "use_aux_skeleton_head": bool(args.use_aux_skeleton_head),
        "prior_dark_quantile": float(args.prior_dark_quantile),
        "prior_kernel": int(args.prior_kernel),
        "prior_clean": not bool(args.no_prior_clean),
        "prior_sigma": float(args.prior_sigma),
        "skeleton_aux_loss_weight": float(args.skeleton_aux_loss_weight),
    }
    best_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for i, batch in enumerate(dataloader):
            x = batch['image']
            target = batch['label']
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()

            aux_skel_logits = None
            if bool(args.use_gs_spade) and bool(args.use_aux_skeleton_head) and float(args.skeleton_aux_loss_weight) > 0.0:
                pred0, pred1, pred2, aux_skel_logits = model(x, return_aux=True)
            else:
                pred0, pred1, pred2 = model(x)

            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2

            if aux_skel_logits is not None and float(args.skeleton_aux_loss_weight) > 0.0:
                skel_gt = build_skeleton_tensor_from_binary_mask(
                    target,
                    clean=not bool(args.no_prior_clean),
                    kernel_size=int(args.prior_kernel),
                ).to(device=aux_skel_logits.device, dtype=aux_skel_logits.dtype)
                if aux_skel_logits.shape[-2:] != skel_gt.shape[-2:]:
                    aux_skel_logits = F.interpolate(
                        aux_skel_logits,
                        size=skel_gt.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                aux_loss = skeleton_dice_loss(aux_skel_logits, skel_gt)
                loss = loss + float(args.skeleton_aux_loss_weight) * aux_loss

            loss.backward()
            optim.step()
            epoch_loss_sum += float(loss.item())
            epoch_steps += 1
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
                
        scheduler.step()
        epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "config": run_config,
            "epoch_loss": epoch_loss,
        }
        latest_path = os.path.join(args.save_dir, "latest_model.pt")
        torch.save(checkpoint, latest_path)
        if epoch_loss <= best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            epoch_path = os.path.join(args.save_dir, f"sa_sam2_epoch_{epoch + 1}.pt")
            torch.save(checkpoint, epoch_path)
            print("[Saving Snapshot:]", epoch_path)


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main(args)
