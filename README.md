# DiffuKnee — Diffusion-based Knee MRI Segmentation

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-active-success)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)

**DiffuKnee** implements a diffusion-model + U-Net pipeline for automated multi-class knee MRI segmentation.  
It provides dataset preparation helpers, training/evaluation workflows, 2D & 3D inference scripts, and reproducibility tools (Docker/conda).

---

## 🚀 Quick links
- **Project name:** DiffuKnee  
- **Task:** Knee MRI multi-class segmentation (NIfTI `.nii` / `.nii.gz`)  
- **Core method:** U-Net backbone + diffusion model for segmentation  

---

## 📚 Table of contents
1. [Highlights](#-highlights)  
2. [Quickstart (TL;DR)](#-quickstart-tldr)  
3. [Detailed setup & usage](#-detailed-setup--usage)  
   - [Install](#install)  
   - [Prepare data](#prepare-data)  
   - [Compute stats & weights](#compute-stats--weights)  
   - [Train](#train)  
   - [Inference (2D / 3D)](#inference-2d--3d)  
   - [Evaluate](#evaluate)  
4. [Configuration example](#-configuration-example)  
5. [Repository structure](#-repository-structure)  
6. [Good-to-have improvements](#-good-to-have-improvements)  
7. [Developer notes & tips](#-developer-notes--tips)  
8. [Contributing, License & Contact](#-contributing-license--contact)  

---

## ✨ Highlights
- Diffusion-guided segmentation for consistent, smooth mask generation.  
- Supports both slice-level (2D) and volume-level (3D) MRI pipelines.  
- Built-in dataset splitters and utilities to compute **mean/std** + **class weights**.  
- Training loop with checkpointing, periodic sampling, and evaluation metrics (Dice/F1, IoU).  
- Reproducible setup with **Dockerfile** and **conda environment.yml**.  

---

## ⚡ Quickstart (TL;DR)

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Create data split (example)
python data/split.py   --raw-dirs /data/raw_images   --mask-dirs /data/masks   --save-dir data/splitted   --test-size 0.2

# 3. Compute mean/std and class weights
python experiments/pretrain.py   --adapt-dir data/splitted   --results-save-dir results/params   --num-classes 6

# 4. Train
python train.py
# or multi-GPU
accelerate launch train.py

# 5. Predict
python predict_2d.py --input /path/to/case.nii.gz     --checkpoint results/checkpoints/best_checkpoint     --output-dir results/predictions     --output-name case001

python predict_3d.py --input /path/to/case.nii.gz     --checkpoint results/checkpoints/best_checkpoint     --output-dir results/predictions     --output-name case001

# 6. Evaluate
python evaluate.py
```

---

## 🛠 Detailed setup & usage

### Install
- Recommended: Python 3.8+ with venv or conda  
- Install dependencies:
```bash
pip install -r requirements.txt
```

### Prepare data
- Input: **NIfTI** (`.nii`, `.nii.gz`) knee MRI volumes and corresponding segmentation masks.  
- Split into `train/`, `train_masks/`, `test/`, `test_masks/` using either:
  ```bash
  python data/split.py --raw-dirs ... --mask-dirs ... --save-dir data/splitted
  ```
  or with a `paths.txt`:
  ```bash
  python data/split_from_paths.py --raw-dirs ... --mask-dirs ... --save-dir data/splitted --paths-file paths.txt
  ```

### Compute stats & class weights
```bash
python experiments/pretrain.py --adapt-dir data/splitted --results-save-dir results/params --num-classes 6
```
Saves `mean_std.pt` and `class_weights.pt` under `results/params/`.

### Train
```bash
python train.py
```
or distributed:
```bash
accelerate launch train.py
```

### Inference (2D / 3D)
```bash
python predict_2d.py --input /path/to/case.nii.gz --checkpoint results/checkpoints/best_checkpoint --output-dir results/predictions --output-name case001
python predict_3d.py --input /path/to/case.nii.gz --checkpoint results/checkpoints/best_checkpoint --output-dir results/predictions --output-name case001
```

### Evaluate
```bash
python evaluate.py
```
Outputs Dice/F1 and IoU scores, also writes `eval.txt`.

---

## 🧾 Configuration example

```yaml
train:
  lr: 5e-5
  epochs: 250
  batch_size: 8
  save_every: 5
  early_stopping_patience: 6
model:
  image_size: 384
  num_classes: 6
paths:
  results: ./results
  checkpoints: ./results/checkpoints
```

---

## 📁 Repository structure
```
DiffuKnee/
├─ data/                # dataset loaders + split helpers
├─ diffusion/           # diffusion model and schedules
├─ unet/                # U-Net backbone & smoothing utilities
├─ experiments/         # trainer + pretrain utilities
├─ results/             # sample outputs, params, examples
├─ utils/               # preprocessing, postprocessing, helper functions
├─ train.py             # main training script
├─ predict_2d.py        # 2D inference example
├─ predict_3d.py        # 3D inference example
├─ evaluate.py          # evaluation & metrics
├─ requirements.txt
├─ Dockerfile
├─ environment.yml
├─ LICENSE
└─ README.md
```

---

## ✅ Good-to-have improvements
- Add a **sample dataset or download script** for demo purposes.  
- Provide `config.yaml` for all training/prediction parameters.  
- Add GitHub **badges** (build, license, Python version).  
- Set up CI/CD (GitHub Actions) for tests & linting.  

---

## 🧩 Developer notes & tips
- Recompute `mean_std.pt` & `class_weights.pt` if dataset changes.  
- Uses **TorchIO** for 3D augmentation.  
- Uses **HuggingFace Accelerate** for multi-GPU/distributed training.  
- Optional **smoothing** and postprocessing in `utils/postprocessing.py`.  

---

## 🤝 Contributing, License & Contact
- Licensed under **MIT** (see `LICENSE`).  
- Contributions welcome! See `CONTRIBUTING.md`.  
- For questions or collaborations, open an issue or pull request on GitHub.  
