#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
import torch
from torchvision import utils
from utils import load_class_weights, load_mean_std
from utils.postprocessing import apply_argmax_and_coloring
from diffusion.gaussian_diffusion import GaussianDiffusion
from unet.unet import Unet
from accelerate import Accelerator
from utils.preprocessing import load_mri, save_mri, smoothing
import torch.nn.functional as F

def load_model(checkpoint_path, device, image_size=384, num_classes=6):
    params_dir = "./results/params"
    mean, std = load_mean_std(params_dir)
    class_weights = load_class_weights(params_dir)
    model = Unet(dim=32, dim_mults=(1,2,4,8,16), norm_mean=mean, norm_std=std)
    diffusion = GaussianDiffusion(model, image_size=image_size, timesteps=100, class_weights=class_weights)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        diffusion.load_state_dict(ckpt.get("model", ckpt))
    diffusion = diffusion.to(device)
    diffusion.eval()
    return diffusion, mean, std

def predict_volume(model, nifti_path, device, batch_size=8):
    img, affine, header = load_mri(nifti_path)
    img = torch.from_numpy(img).unsqueeze(0)  # shape: 1, C, D, H, W or assume caller handles slices
    # This script expects input shape compatible with existing predict logic.
    # For simplicity, follow original repo's approach to select slices.
    img = img.squeeze(0).to(device).float()
    # select middle slice index if 2D prediction
    if img.ndim == 4:  # channels, depth, H, W or depth, H, W?
        # assume shape (channels, depth, H, W)
        depth = img.shape[1]
        slice_idx = depth // 2
        slice_img = img[:, slice_idx, :, :].unsqueeze(0)
    else:
        slice_img = img.unsqueeze(0)
    with torch.no_grad():
        predicted = model.sample(raw=slice_img.to(device).float(), batch_size=1, disable_bar=True, return_all_timesteps=False)
    return predicted.cpu(), affine, header

def main():
    parser = argparse.ArgumentParser(description="Predict knee segmentation for a single NIfTI (2D slice example).")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input NIfTI file path (.nii or .nii.gz)")
    parser.add_argument("--checkpoint", "-c", type=str, required=False, help="Model checkpoint path")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--output-name", type=str, required=True, help="Base name for saved outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device (e.g., cuda:0 or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    model, mean, std = load_model(args.checkpoint, device)
    predicted, affine, header = predict_volume(model, args.input, device)

    # Postprocess and save colored prediction (argmax + coloring)
    colored = apply_argmax_and_coloring(predicted.squeeze(0))
    for i, sample in enumerate(colored):
        utils.save_image(sample / 255.0, output_dir / f\"{args.output_name}_slice_{i}.png\")

    # Also save raw predicted mask as NIfTI if desired
    try:
        pred_mask = torch.argmax(predicted, dim=1).squeeze(0).cpu().numpy()
        save_mri(pred_mask, affine, header, args.output_name + \"_mask.nii.gz\", str(output_dir))
    except Exception as e:
        print(\"Could not save NIfTI mask:\", e)

    print(f\"Saved prediction images to {output_dir}\")


if __name__ == \"__main__\":
    main()
