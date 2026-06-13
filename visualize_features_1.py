"""
visualize_features.py
=====================
提取模型的 768 维 VIT 特征，用 t-SNE / UMAP 降维到 2D，
可视化 7 类婴儿表情在特征空间中的分布。

支持的模型：
  config_c       — Config C（HRNet + Full Unfreeze）
  landmark_bias  — LandmarkBias v1（原始坐标）
  landmark_bias_v2 — LandmarkBias v2（Procrustes + 改进）

使用方法:
  python visualize_features.py \
      --model_type  config_c \
      --img_root    TIF_DB \
      --splits_dir  splits_18fold_130 \
      --lm_dir      predictions \
      --ckpt_dir    results_config_c \
      --out_dir     ./vis_features \
      --method      tsne \
      --gpu 0
"""

import os, sys, csv, argparse, importlib
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from baby_dataset import LABEL_MAP

NUM_CLASSES = 7
IDX2LABEL   = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = [IDX2LABEL[i] for i in range(NUM_CLASSES)]

# 7类表情颜色（色觉友好）
COLORS = [
    '#E24B4A',  # Anger   — 红
    '#BA7517',  # Disgust  — 橙棕
    '#534AB7',  # Fear    — 紫
    '#1D9E75',  # Happy   — 绿
    '#378ADD',  # Neutral — 蓝
    '#D4537E',  # Sad     — 粉
    '#639922',  # Surprise— 黄绿
]

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_landmark_txt_infanface(txt_path):
    """读取 InfAnFace 格式 landmark txt"""
    import re
    with open(txt_path) as f:
        content = f.read()
    x_vals = [float(v) for v in re.findall(
        r'np\.float32\(([\d.]+)\)', content.split('y:')[0])]
    y_vals = [float(v) for v in re.findall(
        r'np\.float32\(([\d.]+)\)', content.split('y:')[1].split('min_box')[0])]
    if len(x_vals) != 68:
        x_vals = x_vals[:68] if len(x_vals) >= 68 else x_vals + [0.0]*(68-len(x_vals))
        y_vals = y_vals[:68] if len(y_vals) >= 68 else y_vals + [0.0]*(68-len(y_vals))
    return np.array(x_vals, np.float32), np.array(y_vals, np.float32)


def load_aligned_txt(txt_path):
    """读取 Procrustes 对齐后的坐标（每行 x y）"""
    xs, ys = [], []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
    return np.array(xs, np.float32), np.array(ys, np.float32)


class UnifiedDataset(Dataset):
    def __init__(self, csv_path, img_root, lm_dir,
                 transform=None, model_type='config_c',
                 lm_reader='infanface'):
        self.img_root   = img_root
        self.lm_dir     = lm_dir
        self.transform  = transform
        self.model_type = model_type
        self.lm_reader  = lm_reader  # 'infanface' or 'aligned'
        self.samples    = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                label = row['Label'].strip()
                if label in LABEL_MAP:
                    self.samples.append((row['File'].strip(), LABEL_MAP[label]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_root, filename)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        lm_path = os.path.join(self.lm_dir,
                               os.path.splitext(filename)[0] + '.txt')
        if os.path.exists(lm_path):
            if self.lm_reader == 'aligned':
                x, y = load_aligned_txt(lm_path)
            else:
                x, y = load_landmark_txt_infanface(lm_path)
                x, y = x / 112.0, y / 112.0
        else:
            x = np.zeros(68, np.float32)
            y = np.zeros(68, np.float32)

        return (img,
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                label, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 特征钩子：提取 VIT 输出的 768 维特征
# ─────────────────────────────────────────────────────────────────────────────
class FeatureHook:
    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        # VIT 输出可能是 [B, num_tokens, 768] 或 [B, 768]
        feat = output.detach().cpu()
        if feat.dim() == 3:
            # [B, num_tokens, 768] → mean pooling → [B, 768]
            feat = feat.mean(dim=1)
        elif feat.dim() == 2:
            # 已经是 [B, 768]，直接用
            pass
        else:
            raise ValueError(f"Unexpected feature dim: {feat.shape}")
        self.features = feat  # [B, 768]

    def register(self, model):
        # 找到 VIT 模块
        base = model.module if hasattr(model, 'module') else model
        base.VIT.register_forward_hook(self.hook_fn)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────
def load_model(args, fold_str, device):
    # 确保 InfAnFace 路径在 sys.path 里（HRNet 需要）
    _infanface_root = '/kaggle/working/Infant-Facial-Landmark-Detection-and-Tracking'
    if _infanface_root not in sys.path:
        sys.path.insert(0, _infanface_root)
    _cwd = os.getcwd()
    os.chdir(_infanface_root)
    try:
        from lib.config import config as _cfg, update_config as _upd
        _upd(_cfg, type('args', (), {
            'cfg': 'experiments/300w/hrnet-r90jt.yaml',
            'model_file': 'infanface_pretrained/hrnet-r90jt.pth'
        })())
    finally:
        os.chdir(_cwd)

    if args.model_type == 'config_c':
        from models.PosterV2_7cls_hrnet import pyramid_trans_expr2
        model = pyramid_trans_expr2(img_size=224, num_classes=NUM_CLASSES)
        ckpt_name = f"configC_{fold_str}_best.pth"
    elif args.model_type == 'landmark_bias':
        mod   = importlib.import_module('models.PosterV2_LandmarkBias')
        model = mod.pyramid_trans_expr2_landmarkbias(img_size=224, num_classes=NUM_CLASSES)
        ckpt_name = f"landmark_bias_{fold_str}_phase2_best.pth"
    elif args.model_type == 'landmark_bias_v2':
        mod   = importlib.import_module('models.PosterV2_LandmarkBias_v2')
        model = mod.pyramid_trans_expr2_landmarkbias_v2(img_size=224, num_classes=NUM_CLASSES)
        ckpt_name = f"landmark_bias_{fold_str}_phase2_best.pth"
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        return None

    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = nn.DataParallel(model).to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 特征提取
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(args, device):
    all_feats   = []
    all_labels  = []
    all_preds   = []

    # lm_reader 决定坐标读取方式
    lm_reader = 'aligned' if 'v2' in args.model_type else 'infanface'

    for fold_id in range(1, 19):
        fold_str = f"fold_{fold_id:02d}"
        test_csv = os.path.join(args.splits_dir, f"{fold_str}_test.csv")
        if not os.path.exists(test_csv):
            continue

        model = load_model(args, fold_str, device)
        if model is None:
            print(f"[SKIP] {fold_str}: checkpoint not found")
            continue

        # 动态 lm_dir（Procrustes 版本需要 fold 子目录）
        if 'v2' in args.model_type:
            lm_dir = os.path.join(args.lm_dir, fold_str)
        else:
            lm_dir = args.lm_dir

        # 注册特征钩子
        hook = FeatureHook().register(model)

        dataset = UnifiedDataset(test_csv, args.img_root, lm_dir,
                                 VAL_TF, args.model_type, lm_reader)
        loader  = DataLoader(dataset, batch_size=8, shuffle=False,
                             num_workers=2, pin_memory=True)

        fold_feats, fold_labels, fold_preds = [], [], []
        with torch.no_grad():
            for imgs, xc, yc, labels, _ in loader:
                imgs = imgs.to(device)
                xc   = xc.to(device)
                yc   = yc.to(device)

                if args.model_type == 'config_c':
                    out = model(imgs)
                else:
                    out = model(imgs, xc, yc)

                # hook.features: [B, 768]，直接 append
                fold_feats.append(hook.features.numpy())
                fold_preds.extend(out.argmax(1).cpu().tolist())
                fold_labels.extend(labels.tolist())

        # stack 成 [N_fold, 768]
        fold_feats_arr = np.concatenate(fold_feats, axis=0)
        all_feats.append(fold_feats_arr)
        all_labels.extend(fold_labels)
        all_preds.extend(fold_preds)
        print(f"  {fold_str}: {len(fold_labels)} samples, feat shape={fold_feats_arr.shape}")

    all_feats  = np.concatenate(all_feats, axis=0)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    return all_feats, all_labels, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────
def plot_tsne(feats_2d, labels, preds, out_path, title=''):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    for ax_idx, (ax, use_pred, subtitle) in enumerate(zip(
        axes,
        [False, True],
        ['Ground Truth', 'Predictions']
    )):
        plot_labels = preds if use_pred else labels
        ax.set_title(subtitle, fontsize=12)

        for cls_id in range(NUM_CLASSES):
            mask = plot_labels == cls_id
            if mask.sum() == 0:
                continue
            ax.scatter(
                feats_2d[mask, 0], feats_2d[mask, 1],
                c=COLORS[cls_id], label=LABEL_NAMES[cls_id],
                s=60, alpha=0.85, edgecolors='white', linewidths=0.4,
            )

        # 标注每类的质心
        for cls_id in range(NUM_CLASSES):
            mask = plot_labels == cls_id
            if mask.sum() == 0:
                continue
            cx = feats_2d[mask, 0].mean()
            cy = feats_2d[mask, 1].mean()
            ax.annotate(
                LABEL_NAMES[cls_id],
                (cx, cy),
                fontsize=9, fontweight='bold',
                color=COLORS[cls_id],
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.6, edgecolor='none'),
                ha='center', va='center',
            )

        ax.set_xlabel('t-SNE dim 1', fontsize=10)
        ax.set_ylabel('t-SNE dim 2', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    # 统一图例
    patches = [mpatches.Patch(color=COLORS[i], label=LABEL_NAMES[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=7,
               fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存：{out_path}")


def plot_confusion_overlay(feats_2d, labels, preds, out_path, title=''):
    """
    叠加显示：正确预测用实心点，错误预测用叉号，
    更直观地看哪些类别容易混淆。
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(title, fontsize=13, fontweight='bold')

    correct_mask = (labels == preds)

    for cls_id in range(NUM_CLASSES):
        cls_mask = labels == cls_id
        # 正确预测：实心圆
        ok = cls_mask & correct_mask
        if ok.sum() > 0:
            ax.scatter(feats_2d[ok, 0], feats_2d[ok, 1],
                       c=COLORS[cls_id], s=60, alpha=0.85,
                       edgecolors='white', linewidths=0.4,
                       label=f"{LABEL_NAMES[cls_id]} ✓",
                       marker='o')
        # 错误预测：叉号
        err = cls_mask & ~correct_mask
        if err.sum() > 0:
            ax.scatter(feats_2d[err, 0], feats_2d[err, 1],
                       c=COLORS[cls_id], s=80, alpha=0.95,
                       edgecolors='black', linewidths=0.8,
                       label=f"{LABEL_NAMES[cls_id]} ✗",
                       marker='X')

    acc = correct_mask.mean() * 100
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.set_title(f"{title}\nAcc={acc:.1f}%  ○=correct  ✗=misclassified",
                 fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=8, ncol=2, frameon=True,
              loc='upper right', framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存：{out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Args & Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_type',  required=True,
                   choices=['config_c', 'landmark_bias', 'landmark_bias_v2'])
    p.add_argument('--img_root',    required=True)
    p.add_argument('--splits_dir',  required=True)
    p.add_argument('--lm_dir',      required=True)
    p.add_argument('--ckpt_dir',    required=True)
    p.add_argument('--out_dir',     default='./vis_features')
    p.add_argument('--method',      default='tsne',
                   choices=['tsne', 'both'])
    p.add_argument('--perplexity',  type=float, default=20,
                   help='t-SNE perplexity（推荐 10-30，样本少用小值）')
    p.add_argument('--gpu',         default='0')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    print(f"模型类型：{args.model_type}")
    print(f"提取特征中...")
    feats, labels, preds = extract_features(args, device)
    print(f"特征矩阵：{feats.shape}，样本数：{len(labels)}")

    # 保存特征供后续分析
    npz_path = os.path.join(args.out_dir, f"{args.model_type}_features.npz")
    np.savez(npz_path, feats=feats, labels=labels, preds=preds,
             label_names=np.array(LABEL_NAMES))
    print(f"特征已保存：{npz_path}")

    # t-SNE 降维
    print(f"\nt-SNE 降维（perplexity={args.perplexity}）...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                random_state=42, max_iter=1000, verbose=1)
    feats_2d = tsne.fit_transform(feats)

    tag = args.model_type

    # 图1：GT vs Pred 并排
    plot_tsne(
        feats_2d, labels, preds,
        os.path.join(args.out_dir, f"{tag}_tsne_gt_vs_pred.png"),
        title=f"t-SNE Feature Distribution — {tag}"
    )

    # 图2：正确/错误预测叠加
    plot_confusion_overlay(
        feats_2d, labels, preds,
        os.path.join(args.out_dir, f"{tag}_tsne_correct_vs_error.png"),
        title=f"Misclassification Overlay — {tag}"
    )

    print("\n完成！")


if __name__ == '__main__':
    main()
