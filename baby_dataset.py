"""
baby_dataset.py
---------------
自定义Dataset，用于从CSV split文件加载婴儿情绪数据集（18折交叉验证）。

CSV 格式:
    Infant, File, Label, Official_Score, Label_Source

目录结构假设:
    <img_root>/
        A02/
            A02F10-JTP-4281AN.jpg
            ...
        A03/
            ...
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# 7类情绪标签映射（与原POSTER++ 7cls保持一致）
LABEL_MAP = {
    'Anger':    0,
    'Disgust':  1,
    'Fear':     2,
    'Happy':    3,
    'Neutral':  4,
    'Sad':      5,
    'Surprise': 6,
}

# ImageNet 归一化参数（与原main.py保持一致）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform():
    """训练集数据增强（与原main.py RAF-DB配置一致）"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])


def get_val_transform():
    """验证/测试集变换（与原main.py一致）"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BabyFaceDataset(Dataset):
    """
    从单个CSV文件（一个fold的train或test split）加载婴儿情绪图像。

    参数
    ----
    csv_path : str
        CSV split文件路径，例如 splits_18fold_130/fold_01_train.csv
    img_root : str
        图像根目录，子目录为婴儿ID（Infant列），文件名为File列
    transform : callable, optional
        torchvision变换流水线
    skip_invalid_score : bool
        若True，跳过Official_Score为 '-' 的样本（标签来源不可靠）
        若False，保留所有有效Label的样本（默认False，保留全部）
    """

    def __init__(self, csv_path, img_root, transform=None, skip_invalid_score=False):
        self.img_root = img_root
        self.transform = transform

        df = pd.read_csv(csv_path)

        # 过滤：去除Label不在LABEL_MAP中的行（如表头残留等）
        df = df[df['Label'].isin(LABEL_MAP.keys())].reset_index(drop=True)

        # 可选：去除Official_Score为 '-' 的样本
        if skip_invalid_score:
            df = df[df['Official_Score'] != '-'].reset_index(drop=True)

        self.samples = df[['Infant', 'File', 'Label']].values.tolist()

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {csv_path}. "
                             f"Check LABEL_MAP and CSV content.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        infant, filename, label_str = self.samples[idx]
        img_path = os.path.join(self.img_root, str(filename))
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Please verify --img_root points to the dataset root directory."
            )

        if self.transform is not None:
            img = self.transform(img)

        label = LABEL_MAP[label_str]
        return img, label

    def get_labels(self):
        """供 ImbalancedDatasetSampler 使用（返回所有样本标签列表）"""
        return [LABEL_MAP[s[2]] for s in self.samples]


def build_fold_loaders(fold_idx, splits_dir, img_root, batch_size,
                       num_workers=4, skip_invalid_score=False,
                       use_imbalanced_sampler=False):
    """
    为指定fold构建train/test DataLoader。

    参数
    ----
    fold_idx : int
        fold序号，1-18
    splits_dir : str
        split CSV文件所在目录（如 ./splits_18fold_130）
    img_root : str
        图像根目录
    batch_size : int
    num_workers : int
    skip_invalid_score : bool
    use_imbalanced_sampler : bool
        若True，训练集使用 ImbalancedDatasetSampler（类似AffectNet-7配置）

    返回
    ----
    train_loader, val_loader, train_dataset, val_dataset
    """
    fold_str = f"fold_{fold_idx:02d}"
    train_csv = os.path.join(splits_dir, f"{fold_str}_train.csv")
    test_csv  = os.path.join(splits_dir, f"{fold_str}_test.csv")

    train_dataset = BabyFaceDataset(
        csv_path=train_csv,
        img_root=img_root,
        transform=get_train_transform(),
        skip_invalid_score=skip_invalid_score,
    )
    val_dataset = BabyFaceDataset(
        csv_path=test_csv,
        img_root=img_root,
        transform=get_val_transform(),
        skip_invalid_score=skip_invalid_score,
    )

    if use_imbalanced_sampler:
        from torchsampler import ImbalancedDatasetSampler
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=ImbalancedDatasetSampler(train_dataset),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset, val_dataset
