#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
import torch
from torchvision import utils
from utils import load_class_weights, load_mean_std
from utils.postprocessing import apply_argmax_and_coloring, save_mri as save_mri_post
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
    img, affine, header = load_mri(nifti_path)  # returns numpy (C, D, H, W) or (D, H, W)
    img = torch.from_numpy(img).unsqueeze(0).to(device).float()  # 1, C, D, H, W or 1, D, H, W
    # If dataset is stored in different shape, adapt accordingly.
    depth = img.shape[2]
    pred_volume = []
    with torch.no_grad():
        # iterate over batches of slices
        for i in range(0, depth, batch_size):
            batch = img[:, :, i : min(i + batch_size, depth), :, :]
            predicted = model.sample(raw=batch.to(device).float(), batch_size=batch.shape[2], disable_bar=True)
            pred_volume.append(predicted.cpu())
    pred_volume = torch.cat(pred_volume, dim=0)
    # reshape/permute to (num_classes, D, H, W) expected by save_mri
    if pred_volume.ndim == 5:
        # (N, C, H, W) stacking logic; adapt to repo's expected shape
        pred_volume = pred_volume.permute(1,0,2,3).contiguous()
    return pred_volume, affine, header

def main():
    parser = argparse.ArgumentParser(description=\"Predict knee segmentation for a 3D NIfTI volume.\")
    parser.add_argument(\"--input\", \"-i\", type=str, required=True, help=\"Input NIfTI file path (.nii or .nii.gz)\")
    parser.add_argument(\"--checkpoint\", \"-c\", type=str, required=False, help=\"Model checkpoint path\")
    parser.add_argument(\"--output-dir\", \"-o\", type=str, required=True, help=\"Directory to save outputs\")
    parser.add_argument(\"--output-name\", type=str, required=True, help=\"Base name for saved outputs\")
    parser.add_argument(\"--device\", type=str, default=(\"cuda\" if torch.cuda.is_available() else \"cpu\"), help=\"Torch device (e.g., cuda:0 or cpu)\")
    parser.add_argument(\"--batch-size\", type=int, default=8, help=\"Batch size for slice-wise sampling\")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    model, mean, std = load_model(args.checkpoint, device)
    pred_volume, affine, header = predict_volume(model, args.input, device, batch_size=args.batch_size)

    # Save argmaxed mask as NIfTI
    try:
        mask = torch.argmax(pred_volume, dim=0).cpu().numpy()  # (D,H,W)
        save_mri(mask, affine, header, args.output_name + \"_mask.nii.gz\", str(output_dir))
    except Exception as e:
        print(\"Failed to save NIfTI:\", e)

    # Save a few PNG slices as thumbnails
    try:
        # convert to colored images for the first 5 slices
        for i in range(min(5, mask.shape[0])):
            slice_mask = torch.from_numpy((mask[i,:,:])).unsqueeze(0)
            img_col = apply_argmax_and_coloring(F.one_hot(slice_mask.long(), num_classes=6).permute(2,0,1).float().unsqueeze(0))
            utils.save_image(img_col[0]/255.0, output_dir / f\"{args.output_name}_slice_{i}.png\")
    except Exception as e:
        print(\"Failed to save PNG slices:\", e)

    print(f\"Saved predictions to {output_dir}\")


if __name__ == \"__main__\":
    main()
