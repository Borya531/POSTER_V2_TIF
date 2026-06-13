# POSTER V2 — Infant Emotion Recognition (TIF Dataset, 18-fold CV)

This repository adapts **[POSTER V2](https://arxiv.org/abs/2301.12149)** — a state-of-the-art Facial Expression Recognition (FER) network — for **infant emotion recognition** using the TIF (Tobii Infant Face) dataset, with HRNet landmark features replacing the original MobileFaceNet landmark backbone.

## Overview

POSTER V2 fuses facial landmark features with image appearance features through window-based cross-attention. This fork replaces the adult-face MobileFaceNet landmark extractor with **HRNet-R90JT** pre-trained on infant faces (via the [Infant-Facial-Landmark-Detection-and-Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) project), enabling better landmark localisation on infant faces.

**Key additions over the original POSTER V2:**

| File | Description |
|---|---|
| `hrnet_landmark_backbone.py` | Drop-in HRNet replacement for MobileFaceNet; adapts 4 HRNet stage-4 branches to POSTER V2's expected feature dimensions |
| `baby_dataset.py` | Dataset class for the 18-fold TIF cross-validation splits (CSV format: `Infant, File, Label, Official_Score, Label_Source`) |
| `main_config_c.py` | Training entry-point for Config C — HRNet landmark backbone + fully unfrozen IR50 |
| `tif_annotated_130.csv` | Full annotation file for 130 TIF images (7 emotion classes) |
| `splits_18fold_130/` | Pre-generated 18-fold stratified splits (`fold_NN_train.csv`, `fold_NN_test.csv`) |
| `data_preprocessing/sam.py` | SAM (Sharpness-Aware Minimisation) optimiser implementation |

## Project Structure

```
POSTER_V2_TIF/
├── main.py                    # Original POSTER V2 train/eval (RAF-DB, AffectNet, CAER-S)
├── main_8.py                  # 8-class AffectNet variant
├── main_config_c.py           # TIF 18-fold training — Config C (HRNet + full IR50)
├── baby_dataset.py            # TIF infant emotion dataset loader
├── hrnet_landmark_backbone.py # HRNet → POSTER V2 adapter module
├── visualize_features_1.py    # Feature map visualisation utilities
├── tif_annotated_130.csv      # Ground-truth labels for 130 TIF images
├── requirements.txt           # Python dependencies
├── models/
│   ├── PosterV2_7cls.py       # POSTER V2 (7-class, MobileFaceNet landmarks)
│   ├── PosterV2_7cls_hrnet.py # POSTER V2 (7-class, HRNet landmarks) ← main variant
│   ├── PosterV2_8cls.py       # POSTER V2 (8-class AffectNet variant)
│   ├── ir50.py                # IR-50 image backbone
│   ├── vit_model.py           # Vision Transformer cross-attention (7-class)
│   ├── vit_model_8.py         # Vision Transformer cross-attention (8-class)
│   ├── mobilefacenet.py       # Original MobileFaceNet landmark backbone
│   ├── matrix.py              # Confusion matrix utilities
│   └── pretrain/              # ← Place downloaded weights here (see below)
├── splits_18fold_130/         # 18 × (train + test) CSV splits
│   ├── fold_01_train.csv
│   ├── fold_01_test.csv
│   └── ...
├── data_preprocessing/
│   └── sam.py                 # SAM optimiser
├── figures/
│   └── fig1.png               # Architecture diagram
├── checkpoint/                # Training checkpoints (git-ignored)
└── log/                       # TensorBoard logs (git-ignored)
```

## Requirements

- Python ≥ 3.8
- PyTorch 1.8.1 + CUDA 11.1 (see `requirements.txt` for full pinned deps)
- The **Infant-Facial-Landmark-Detection-and-Tracking** repo (for HRNet weights and config)

```bash
pip install -r requirements.txt
```

## Pretrained Weights

Download and place under `models/pretrain/`:

| Weight | Source |
|--------|--------|
| `ir50.pth` | [Google Drive](https://drive.google.com/file/d/17QAIPlpZUwkQzOTNiu-gUFLTqAxS-qHt/view?usp=sharing) |
| `mobilefacenet_model_best.pth.tar` | [Google Drive](https://drive.google.com/file/d/1SMYP5NDkmDE3eLlciN7Z4px-bvFEuHEX/view?usp=sharing) |
| `hrnet-r90jt.pth` (infant HRNet) | From [Infant-Facial-Landmark-Detection-and-Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) |

## Data Setup

### TIF Dataset (infant emotion, 18-fold CV)

Organise TIF images by infant ID:

```
tif_images/
    A02/
        A02F10-JTP-4281AN.jpg
        ...
    A03/
        ...
```

The annotation file and cross-validation splits are already included in this repo (`tif_annotated_130.csv`, `splits_18fold_130/`).

### Standard FER Datasets (RAF-DB / AffectNet / CAER-S)

Follow the [original POSTER V2 data preparation instructions](https://github.com/Talented-Q/POSTER_V2#preparation).

## Training

### TIF — Config C (HRNet + fully unfrozen IR50)

```bash
python main_config_c.py \
    --img_root      /path/to/tif_images \
    --splits_dir    ./splits_18fold_130 \
    --infanface_root /path/to/Infant-Facial-Landmark-Detection-and-Tracking \
    --hrnet_pth     /path/to/hrnet-r90jt.pth \
    --out_dir       ./results_config_c \
    --epochs        200 \
    --batch_size    64 \
    --lr            3.5e-5 \
    --gpu           0
```

### Standard POSTER V2 training

```bash
# RAF-DB (7 cls)
python main.py --data /path/to/raf-db --data_type RAF-DB \
               --lr 3.5e-5 --batch-size 144 --epochs 200 --gpu 0

# AffectNet (7 cls)
python main.py --data /path/to/affectnet-7 --data_type AffectNet-7 \
               --lr 1e-6 --batch-size 144 --epochs 200 --gpu 0

# AffectNet (8 cls)
python main_8.py --data /path/to/affectnet-8 \
                 --lr 1e-6 --batch-size 144 --epochs 200 --gpu 0
```

## Evaluation

```bash
# TIF 18-fold cross-validation
python main_config_c.py \
    --img_root /path/to/tif_images \
    --evaluate /path/to/checkpoint.pth \
    --gpu 0

# Standard FER datasets
python main.py --data /path/to/dataset --evaluate /path/to/checkpoint.pth
```

## Cross-Validation Splits

The `splits_18fold_130/` directory contains 18 pre-generated stratified splits of the 130-image TIF dataset. Each fold file has columns:

```
Infant, File, Label, Official_Score, Label_Source
```

Emotion classes: `Anger`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`

## Architecture Notes

### HRNet Landmark Backbone (`hrnet_landmark_backbone.py`)

HRNet stage-4 produces 4 feature branches at different resolutions. This module maps them to the 3 feature tensors POSTER V2 expects from MobileFaceNet via learnable 1×1 conv adapters:

| POSTER V2 expects | HRNet source | Adapter output |
|---|---|---|
| `[B, 64, 28, 28]` | branch 1 `[B, 18, 64, 64]` | → `[B, 64, 28, 28]` |
| `[B, 128, 14, 14]` | branch 2 `[B, 36, 32, 32]` | → `[B, 128, 14, 14]` |
| `[B, 512, 7, 7]` | branch 4 `[B, 144, 8, 8]` | → `[B, 512, 7, 7]` |

The HRNet weights are optionally frozen; only the adapters are trained by default.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements & Citation

This project builds on [POSTER V2](https://github.com/Talented-Q/POSTER_V2) and the [Infant Facial Landmark Detection and Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) codebase.

If you use this code, please cite the original POSTER V2 paper:

```bibtex
@article{mao2023poster,
  title={POSTER V2: A simpler and stronger facial expression recognition network},
  author={Mao, Jiawei and Xu, Rui and Yin, Xuesong and Chang, Yuanqi and Nie, Binling and Huang, Aibin},
  journal={arXiv preprint arXiv:2301.12149},
  year={2023}
}
```
