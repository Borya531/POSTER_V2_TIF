"""
main_config_c.py — Config C: HRNet + Full Model Unfrozen
=========================================================
Config C: HRNet (infant coordinates) + Full IR50 Unfrozen
  - Landmark:  HRNetLandmarkBackbone (freeze_hrnet=True, adapters trainable)
  - IR50:      全部解冻（input_layer + body1 + body2 + body3）
  - LR:        统一 lr=3.5e-5，无 scheduler
  - 对比目标:  与 Config B (Freeze Body1) 对比，
               测试完整 IR50 重训练的效果

使用方法（在 POSTER_V2-main/ 目录下）:
  python main_config_c.py \
      --img_root      /path/to/tif_images \
      --splits_dir    /path/to/splits_18fold_130 \
      --infanface_root /path/to/Infant-Facial-Landmark-Detection-and-Tracking \
      --out_dir       ./results_config_c \
      --gpu           0
"""

import warnings
import os
import sys
import argparse
import datetime
import numpy as np

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from data_preprocessing.sam import SAM
from baby_dataset import build_fold_loaders, LABEL_MAP

warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLASSES = 7
IDX2LABEL   = {v: k for k, v in LABEL_MAP.items()}
now         = datetime.datetime.now()
time_str    = now.strftime("[%m-%d]-[%H-%M]")

# ─── 参数解析 ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Config B: HRNet + Freeze Body1')
parser.add_argument('--img_root',       type=str, required=True)
parser.add_argument('--splits_dir',     type=str, default='./splits_18fold_130')
parser.add_argument('--out_dir',        type=str, default='./results_config_c')
parser.add_argument('--infanface_root', type=str,
                    default='/kaggle/working/Infant-Facial-Landmark-Detection-and-Tracking',
                    help='InfAnFace 项目根目录')
parser.add_argument('--hrnet_cfg',      type=str,
                    default='experiments/300w/hrnet-r90jt.yaml',
                    help='HRNet yaml 配置（相对于 infanface_root）')
parser.add_argument('--hrnet_pth',      type=str,
                    default='infanface_pretrained/hrnet-r90jt.pth',
                    help='HRNet 预训练权重（相对于 infanface_root）')
parser.add_argument('--num_folds',      type=int,   default=18)
parser.add_argument('--start_fold',     type=int,   default=1)
parser.add_argument('--epochs',         type=int,   default=200)
parser.add_argument('--batch_size',     type=int,   default=64)
parser.add_argument('--lr',             type=float, default=3.5e-5)
parser.add_argument('--workers',        type=int,   default=4)
parser.add_argument('--print_freq',     type=int,   default=30)
parser.add_argument('--gpu',            type=str,   default='0')
parser.add_argument('--skip_invalid_score', action='store_true')
args = parser.parse_args()


# ─── 加载 HRNet 版模型 ───────────────────────────────────────────────────────
sys.path.insert(0, args.infanface_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.PosterV2_7cls_hrnet import pyramid_trans_expr2
print(f'[Config B] HRNet backbone  cfg={args.hrnet_cfg}  pth={args.hrnet_pth}')


# ─── Config B 冻结策略 ───────────────────────────────────────────────────────
def apply_config_c(model: nn.Module) -> nn.Module:
    """
    在 DataParallel 包装之前调用。
    face_landback.hrnet:           freeze_hrnet=True 已在模型内部冻结
    face_landback.adapter1/2/3:    解冻
    ir_back 全部:                   解冻（input_layer + body1 + body2 + body3）
    所有 Transformer/适配器层:      解冻
    """
    # Step 1：全部冻结
    for param in model.parameters():
        param.requires_grad = False

    # Step 2：解冻整个 IR50
    for param in model.ir_back.parameters():
        param.requires_grad = True

    # Step 3：解冻 Transformer/适配器（包含 window LayerNorm）
    for m in [
        model.attn1,  model.attn2,  model.attn3,
        model.ffn1,   model.ffn2,   model.ffn3,
        model.conv1,  model.conv2,  model.conv3,
        model.window1, model.window2, model.window3,
        model.embed_q, model.embed_k, model.embed_v,
        model.last_face_conv,
        model.VIT,
    ]:
        for param in m.parameters():
            param.requires_grad = True

    # Step 4：HRNet adapter 解冻
    for m in [model.face_landback.adapter1,
              model.face_landback.adapter2,
              model.face_landback.adapter3]:
        for param in m.parameters():
            param.requires_grad = True

    _print_param_summary(model)
    return model


def _print_param_summary(model: nn.Module):
    total, trainable = 0, 0
    groups: dict = {}
    for name, param in model.named_parameters():
        total += param.numel()
        parts  = name.split('.')
        if parts[0] == 'ir_back':
            key = f"ir_back.{parts[1]}"
        elif parts[0] == 'face_landback':
            key = f"face_landback.{parts[1]}"
        else:
            key = parts[0]
        groups.setdefault(key, {'total': 0, 'trainable': 0})
        groups[key]['total'] += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            groups[key]['trainable'] += param.numel()

    print(f"\n{'─'*62}")
    print(f"  {'Module':<32} {'Trainable':>12} {'Total':>12}")
    print(f"{'─'*62}")
    for g, v in sorted(groups.items()):
        s = f"{v['trainable']:,}" if v['trainable'] > 0 else '(frozen)'
        print(f"  {g:<32} {s:>12} {v['total']:>12,}")
    pct = 100.0 * trainable / total if total else 0
    print(f"{'─'*62}")
    print(f"  {'Total':<32} {trainable:>12,} {total:>12,}  ({pct:.1f}% trainable)\n")


# ─── 辅助类 ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


class RecorderMeter:
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.epoch_losses   = np.zeros((total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((total_epoch, 2), dtype=np.float32)
    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx]   = [train_loss * 30, val_loss * 30]
        self.epoch_accuracy[idx] = [train_acc, val_acc]
    def plot_curve(self, save_path):
        fig, ax = plt.subplots(figsize=(18, 8), dpi=80)
        x = np.arange(self.total_epoch)
        ax.plot(x, self.epoch_accuracy[:, 0], 'g-',  label='train-acc')
        ax.plot(x, self.epoch_accuracy[:, 1], 'y-',  label='val-acc')
        ax.plot(x, self.epoch_losses[:, 0],   'g--', label='train-loss×30')
        ax.plot(x, self.epoch_losses[:, 1],   'y--', label='val-loss×30')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
        ax.grid(True); ax.legend()
        fig.savefig(save_path, bbox_inches='tight'); plt.close(fig)


def _accuracy(output, target):
    with torch.no_grad():
        return (output.argmax(dim=1) == target).float().mean().item() * 100


def _log(msg, path):
    print(msg)
    with open(path, 'a') as f:
        f.write(msg + '\n')


# ─── Train / Validate ────────────────────────────────────────────────────────

def train_one_epoch(loader, model, criterion, optimizer, epoch, log_path):
    losses = AverageMeter()
    acc_m  = AverageMeter()
    model.train()

    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(), target.cuda()

        # SAM first step
        output = model(images)
        loss   = criterion(output, target)
        losses.update(loss.item(), images.size(0))
        acc_m.update(_accuracy(output, target), images.size(0))
        optimizer.zero_grad(); loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM second step
        output = model(images)
        loss   = criterion(output, target)
        losses.update(loss.item(), images.size(0))
        acc_m.update(_accuracy(output, target), images.size(0))
        optimizer.zero_grad(); loss.backward()
        optimizer.second_step(zero_grad=True)

        if i % args.print_freq == 0:
            _log(f"  [{epoch+1}][{i}/{len(loader)}] "
                 f"loss={losses.avg:.4f}  acc={acc_m.avg:.1f}%", log_path)

    return acc_m.avg, losses.avg


def validate(loader, model, criterion, log_path):
    losses = AverageMeter()
    all_preds, all_targets = [], []
    model.eval()

    with torch.no_grad():
        for images, target in loader:
            images, target = images.cuda(), target.cuda()
            output = model(images)
            losses.update(criterion(output, target).item(), images.size(0))
            all_preds.extend(output.argmax(dim=1).cpu().tolist())
            all_targets.extend(target.cpu().tolist())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc         = float((all_preds == all_targets).mean() * 100)
    cm          = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
    per_cls_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1) * 100

    _log(f" **** Val Accuracy: {acc:.3f}% ****", log_path)
    return acc, losses.avg, cm, per_cls_acc, all_preds, all_targets


def _plot_confusion_matrix(cm, save_dir, fold_idx):
    labels = [IDX2LABEL[i] for i in range(NUM_CLASSES)]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted', ylabel='True',
           title=f'Fold {fold_idx:02d} - Normalized Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            ax.text(c, r, f'{cm_norm[r,c]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[r,c] > 0.5 else 'black', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=100)
    plt.close(fig)


# ─── 单 fold 训练 ─────────────────────────────────────────────────────────────

def train_one_fold(fold_idx):
    fold_str     = f"fold_{fold_idx:02d}"
    fold_dir     = os.path.join(args.out_dir, fold_str)
    os.makedirs(fold_dir, exist_ok=True)
    log_path     = os.path.join(fold_dir, 'log.txt')
    ckpt_path    = os.path.join(args.out_dir, f"configC_{fold_str}_best.pth")

    now_str = datetime.datetime.now().strftime("%m-%d %H:%M")
    _log(f"\n{'='*60}", log_path)
    _log(f"  Config C  Fold {fold_idx:02d}/{args.num_folds}   [{now_str}]", log_path)
    _log(f"  HRNet cfg={args.hrnet_cfg}", log_path)
    _log(f"  LR={args.lr}  BS={args.batch_size}  Epochs={args.epochs}", log_path)
    _log(f"{'='*60}", log_path)

    # ── 数据 ─────────────────────────────────────────────────
    train_loader, val_loader, train_ds, val_ds = build_fold_loaders(
        fold_idx           = fold_idx,
        splits_dir         = args.splits_dir,
        img_root           = args.img_root,
        batch_size         = args.batch_size,
        num_workers        = args.workers,
        skip_invalid_score = args.skip_invalid_score,
    )
    _log(f"  Train: {len(train_ds)} | Test: {len(val_ds)}", log_path)

    # ── 模型：每个 fold 重新初始化，从相同预训练权重出发 ─────
    model = pyramid_trans_expr2(img_size=224, num_classes=NUM_CLASSES)
    model = apply_config_c(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = SAM(
        [p for p in model.parameters() if p.requires_grad],
        torch.optim.Adam, lr=args.lr, rho=0.05, adaptive=False
    )
    _log(f"  [Uniform LR] lr={args.lr:.2e}, no scheduler", log_path)

    recorder   = RecorderMeter(args.epochs)
    best_acc   = 0.0
    best_result = {}

    for epoch in range(args.epochs):
        lr_now = optimizer.state_dict()['param_groups'][0]['lr']
        _log(f"\n  Epoch [{epoch+1}/{args.epochs}]  lr={lr_now:.2e}", log_path)

        train_acc, train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, log_path)
        val_acc, val_loss, cm, per_cls, preds, targets = validate(
            val_loader, model, criterion, log_path)

        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
        recorder.plot_curve(os.path.join(fold_dir, 'curve.png'))

        _log(f"  train acc={train_acc:.1f}%  loss={train_loss:.4f}", log_path)
        _log(f"  test  acc={val_acc:.1f}%   loss={val_loss:.4f}", log_path)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if is_best:
            best_result = {
                'fold': fold_idx, 'best_acc': best_acc,
                'cm': cm, 'per_cls_acc': per_cls,
                'preds': preds, 'targets': targets,
            }
            torch.save({
                'fold':       fold_idx,
                'epoch':      epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc':   best_acc,
                'config':     'C',
            }, ckpt_path)
            _log(f"  *** New best {best_acc:.1f}% → {ckpt_path}", log_path)

    # fold 结束汇总
    if best_result:
        np.save(os.path.join(fold_dir, 'best_confusion_matrix.npy'),
                best_result['cm'])
        _plot_confusion_matrix(best_result['cm'], fold_dir, fold_idx)
        cls_str = '  '.join(
            f"{IDX2LABEL[i]}={best_result['per_cls_acc'][i]:.0f}%"
            for i in range(NUM_CLASSES)
        )
        _log(f"\n  Fold {fold_idx:02d} best: {best_acc:.1f}%  [{cls_str}]", log_path)

    return float(best_acc), best_result


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    log_path     = os.path.join(args.out_dir, f"configC-{time_str}-log.txt")
    summary_path = os.path.join(args.out_dir, f"configC-{time_str}-summary.txt")

    _log("Config C: HRNet + Full Model Unfrozen", log_path)
    _log(f"Start:       {now.strftime('%Y-%m-%d %H:%M')}", log_path)
    _log(f"img_root:    {args.img_root}", log_path)
    _log(f"splits_dir:  {args.splits_dir}", log_path)
    _log(f"infanface:   {args.infanface_root}", log_path)
    _log(f"Epochs={args.epochs}  BS={args.batch_size}  LR={args.lr}", log_path)

    all_accs    = []
    all_results = []

    for fold_idx in range(args.start_fold, args.num_folds + 1):
        best_acc, result = train_one_fold(fold_idx)
        all_accs.append(best_acc)
        if result:
            all_results.append(result)

        # 实时更新 summary
        with open(summary_path, 'w') as f:
            f.write("Config C: HRNet + Full Model Unfrozen\n")
            f.write("=" * 40 + "\n")
            for i, acc in enumerate(all_accs, start=args.start_fold):
                f.write(f"Fold {i:02d}: {acc:.4f}%\n")
            if all_accs:
                f.write(f"\nMean: {np.mean(all_accs):.4f}%\n")
                f.write(f"Std:  {np.std(all_accs):.4f}%\n")

    # 全局汇总
    mean_acc = float(np.mean(all_accs))
    std_acc  = float(np.std(all_accs))

    total_cm      = sum(r['cm'] for r in all_results)
    per_cls_global = total_cm.diagonal() / total_cm.sum(axis=1).clip(min=1) * 100
    all_preds   = np.concatenate([r['preds']   for r in all_results])
    all_targets = np.concatenate([r['targets'] for r in all_results])
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    uar      = float(per_cls_global.mean())

    _log(f"\n{'='*60}", log_path)
    _log(f"  Config C — {len(all_accs)}-Fold CV Summary", log_path)
    _log(f"{'='*60}", log_path)
    _log(f"  Mean Acc : {mean_acc:.2f}% ± {std_acc:.2f}%", log_path)
    _log(f"  UAR      : {uar:.2f}%", log_path)
    _log(f"  Macro F1 : {macro_f1:.2f}%", log_path)
    _log(f"\n  Per-class accuracy (aggregated):", log_path)
    for i in range(NUM_CLASSES):
        _log(f"    {IDX2LABEL[i]:<10} {per_cls_global[i]:.1f}%", log_path)
    _log(f"\n  Per-fold accuracy:", log_path)
    for i, acc in enumerate(all_accs, start=args.start_fold):
        _log(f"    fold_{i:02d}  {acc:.1f}%", log_path)
    _log(f"\n  Confusion Matrix (summed):", log_path)
    _log(f"  {[IDX2LABEL[i] for i in range(NUM_CLASSES)]}", log_path)
    _log(str(total_cm), log_path)

    npz_path = os.path.join(args.out_dir, f"configC-{time_str}-results.npz")
    np.savez(npz_path,
             fold_accs        = np.array(all_accs),
             mean_acc         = mean_acc,
             std_acc          = std_acc,
             uar              = uar,
             macro_f1         = macro_f1,
             confusion_matrix = total_cm,
             per_cls_acc      = per_cls_global,
             all_preds        = all_preds,
             all_targets      = all_targets,
             label_names      = np.array([IDX2LABEL[i] for i in range(NUM_CLASSES)]))
    _log(f"\n  Results saved → {npz_path}", log_path)
    _log(f"{'='*60}", log_path)

    print(f'\n{"="*60}')
    print(f'Config C Final: Mean={mean_acc:.2f}% ± {std_acc:.2f}%')
    print(f'Results: {args.out_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
