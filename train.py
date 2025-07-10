import os
import argparse
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from datagen import *
from network import Network
import torch.nn.functional as F



def compute_weighted_bce_iou_loss(prediction, target):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(target, 31, 1, 15) - target)
    bce_map = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
    weighted_bce = (weight * bce_map).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    
    prediction = torch.sigmoid(prediction)
    intersection = ((prediction * target) * weight).sum(dim=(2, 3))
    union = ((prediction + target) * weight).sum(dim=(2, 3))
    weighted_iou = 1 - (intersection + 1) / (union - intersection + 1)

    return (weighted_bce + weighted_iou).mean()


def calculate_dice_numpy(pred, target, eps=1e-6):
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    overlap = np.logical_and(pred_bin, target_bin).sum()
    return (2. * overlap + eps) / (pred_bin.sum() + target_bin.sum() + eps)


def batch_dice(pred_logits, target_masks):
    preds = torch.sigmoid(pred_logits).detach().cpu().numpy()
    gts = target_masks.detach().cpu().numpy()
    return np.mean([calculate_dice_numpy(p[0], g[0]) for p, g in zip(preds, gts)])


def run_training(config):
    train_set = DataGen(config.train_image_path, config.train_mask_path, 352)
    val_set = DataGen(config.test_image_path, config.test_mask_path, 352)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    device = torch.device("cuda")
    model = Network(config.hiera_path).to(device)  
    model.load_state_dict(torch.load(r'C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2unet\weights\proposed\base\x\weights\best_model.pth'), strict=True)
    print('loaded X model')

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, config.epoch, eta_min=1e-7)

    os.makedirs(config.save_path, exist_ok=True)
    best_dice = 0.0

    for epoch_idx in range(config.epoch):
        model.train()
        cumulative_loss = 0.0
        epoch_train_dice = []

        for step, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['label'].to(device).float()

            optimizer.zero_grad()
            out_main, out_aux1, out_aux2 = model(images)

            loss_main = compute_weighted_bce_iou_loss(out_main, masks)
            loss_aux1 = compute_weighted_bce_iou_loss(out_aux1, masks)
            loss_aux2 = compute_weighted_bce_iou_loss(out_aux2, masks)

            total_loss = loss_main + loss_aux1 + loss_aux2
            total_loss.backward()
            optimizer.step()

            cumulative_loss += total_loss.item()
            epoch_train_dice.append(batch_dice(out_main, masks))

            if step % 50 == 0:
                print(f"Epoch {epoch_idx+1} | Step {step+1} | Loss: {total_loss.item():.4f} | Dice: {epoch_train_dice[-1]:.4f}")

        scheduler.step()
        avg_train_dice = np.mean(epoch_train_dice)
        print(f"[Train] Epoch {epoch_idx+1} | Total Loss: {cumulative_loss:.4f} | Mean Dice: {avg_train_dice:.4f}")

        # === Validation ===
        model.eval()
        val_dice_scores = []
        with torch.no_grad():
            for val_batch in val_loader:
                val_img = val_batch['image'].to(device)
                val_mask = val_batch['label'].to(device).float()
                val_pred, _, _ = model(val_img)
                val_dice_scores.append(batch_dice(val_pred, val_mask))

        avg_val_dice = np.mean(val_dice_scores)
        print(f"[Val] Epoch {epoch_idx+1} | Avg Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(config.save_path, 'best_model.pth'))
            print(f"[Best Model Saved @ Epoch {epoch_idx+1}] Dice: {avg_val_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Network")
    parser.add_argument("--hiera_path", type=str, default = r"C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2unet\sam2_hiera_large.pt", required=False, 
                        help="path to the sam2 pretrained hiera")
    parser.add_argument("--train_image_path", type=str, default = r'D:\PhD\Prostate\Data\samunet\uab\train\images', required=False, 
                        help="path to the image that used to train the model")
    parser.add_argument("--train_mask_path", type=str, default = r'D:\PhD\Prostate\Data\samunet\uab\train\masks', required=False,
                        help="path to the mask file for training")
    parser.add_argument("--test_image_path", type=str, default = r'D:\PhD\Prostate\Data\samunet\uab\test\images', required=False, 
                        help="path to the image that used to train the model")
    parser.add_argument("--test_mask_path", type=str, default = r'D:\PhD\Prostate\Data\samunet\uab\test\masks', required=False,
                        help="path to the mask file for training")
    parser.add_argument('--save_path', type=str, default = r'C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2unet\weights\proposed\tl\uab_on_xnn\weights', required=False,
                        help="path to store the checkpoint")
    parser.add_argument("--epoch", type=int, default=20, 
                        help="training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    args = parser.parse_args()

    run_training(args)
