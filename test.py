import argparse
import os
import torch
import numpy as np
import imageio
import torch.nn.functional as F
from network import Network
from datagen import *
import torch.nn.functional as F

def compute_dice(pred_mask, true_mask, eps=1e-6):
    """Compute binary Dice score between prediction and ground truth."""
    pred_binary = pred_mask > 0.5
    true_binary = true_mask > 0.5
    intersection = np.logical_and(pred_binary, true_binary).sum()
    return (2. * intersection + eps) / (pred_binary.sum() + true_binary.sum() + eps)

def run_inference(config):
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = DataGen(config.test_images, config.test_labels, 352, mode='test')
    model = Network().to(device)
    model.load_state_dict(torch.load(config.checkpoint), strict=True)
    model.eval()

    all_dice_scores = []

    for idx in range(len(test_dataset)):
        with torch.no_grad():
            sample = test_dataset[idx]
            input_tensor = sample['image'].unsqueeze(0).to(device)
            label_np = sample['label'].squeeze().numpy()
            filename = sample['name']
            label_np = np.asarray(label_np, dtype=np.float32)
            input_tensor = input_tensor.to(device)
            prediction, _, _ = model(input_tensor)
            prediction = F.interpolate(prediction, size=(128, 128), mode='bilinear', align_corners=False)
            prediction = torch.sigmoid(prediction).cpu().numpy().squeeze()
    

            # Normalize to [0, 255]
            norm_pred = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            #norm_pred = (norm_pred * 255).astype(np.uint8)
            norm_pred = (norm_pred > 0.5).astype(np.uint8) * 255

            output_path = os.path.join(config.output_dir, filename)
            imageio.imsave(output_path, norm_pred)

            print(f"[{idx+1}/{len(test_dataset)}] Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2-UNet Inference Script")
    parser.add_argument("--checkpoint", type=str, default=r"C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2unet\weights\proposed\tl\uab_on_xnn\weights\best_model.pth", help="Path to trained checkpoint")
    parser.add_argument("--test_images", type=str, default=r'D:\PhD\Prostate\Data\samunet\uab\test\images', help="Path to test images")
    parser.add_argument("--test_labels", type=str, default=r'D:\PhD\Prostate\Data\samunet\uab\test\masks', help="Path to test masks")
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\UAB\CK_WorkPlace\PhD\Prostate\sam2unet\weights\proposed\tl\uab_on_xnn\test", help="Directory to save predicted masks")
    args = parser.parse_args()

    run_inference(args)
