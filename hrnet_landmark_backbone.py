"""
hrnet_landmark_backbone.py
===========================
将 HRNet-R90JT 封装成与 MobileFaceNet 完全兼容的 drop-in 替换模块。

MobileFaceNet 在 POSTER++ 中的 forward() 返回：
    out3          [B, 64,  28, 28]   → query for attn1 (window_size=28)
    out4          [B, 128, 14, 14]   → query for attn2 (window_size=14)
    conv_features [B, 512,  7,  7]   → query for attn3 (window_size=7)
                                       经过 last_face_conv 压缩为 [B, 256, 7, 7]

HRNet stage4 输出四个分支（来自 stage4 的 x_fuse）：
    x[0]  [B, 18,  64, 64]   (最高分辨率)
    x[1]  [B, 36,  32, 32]
    x[2]  [B, 72,  16, 16]
    x[3]  [B, 144,  8,  8]   (最低分辨率)

本模块在 head (heatmap层) 之前截取这四个分支，
用可学习的 1x1 卷积 adapter 映射到 MobileFaceNet 所需的通道数和空间尺寸。

使用方法（替换 PosterV2_7cls.py 中的 MobileFaceNet）：
    from hrnet_landmark_backbone import HRNetLandmarkBackbone
    self.face_landback = HRNetLandmarkBackbone(config, pretrained_path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


class HRNetLandmarkBackbone(nn.Module):
    """
    Drop-in 替换 MobileFaceNet，输出与其完全相同的三个特征图。

    forward(x) 输入:  [B, 3, 112, 112]  (POSTER++ 已做 interpolate)
    forward(x) 输出:
        feat1  [B, 64,  28, 28]
        feat2  [B, 128, 14, 14]
        feat3  [B, 512,  7,  7]
    """

    def __init__(self, hrnet_config, pretrained_path: str,
                 freeze_hrnet: bool = True):
        """
        Args:
            hrnet_config:    InfAnFace 的 config 对象（从 yaml 加载）
            pretrained_path: hrnet-r90jt.pth 的路径
            freeze_hrnet:    是否冻结 HRNet 权重（与 MobileFaceNet 原始设置一致）
        """
        super().__init__()

        # ── 1. 加载 HRNet 主干 ──────────────────────────────────────────
        # 需要把 InfAnFace 项目路径加入 sys.path
        hrnet_root = os.environ.get('INFANFACE_ROOT',
                                    '/kaggle/working/Infant-Facial-Landmark-Detection-and-Tracking')
        if hrnet_root not in sys.path:
            sys.path.insert(0, hrnet_root)

        from lib.models.hrnet import HighResolutionNet

        # 不初始化 imagenet 预训练，直接用 infanface 权重
        hrnet_config.defrost()
        hrnet_config.MODEL.INIT_WEIGHTS = False
        hrnet_config.freeze()

        self.hrnet = HighResolutionNet(hrnet_config)

        # 加载 infanface 预训练权重
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        # 去除 DataParallel 的 'module.' 前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing, unexpected = self.hrnet.load_state_dict(state_dict, strict=False)
        print(f'[HRNetLandmarkBackbone] Loaded weights | '
              f'missing: {len(missing)} | unexpected: {len(unexpected)}')

        # 冻结 HRNet（与原 MobileFaceNet 设置一致：requires_grad=False）
        if freeze_hrnet:
            for param in self.hrnet.parameters():
                param.requires_grad = False
            print('[HRNetLandmarkBackbone] HRNet weights frozen.')

        # ── 2. HRNet stage4 各分支通道数 ──────────────────────────────
        # 来自 yaml: NUM_CHANNELS: [18, 36, 72, 144]，BLOCK: BASIC（expansion=1）
        hrnet_channels = [18, 36, 72, 144]   # x[0..3] 的通道数

        # ── 3. Adapter：将 HRNet 特征映射到 MobileFaceNet 等价尺寸 ────
        #
        # 目标输出          来源分支          操作
        # [B,  64, 28, 28]  x[1] [B,36,32,32]  1x1 conv(36→64)  + bilinear→28
        # [B, 128, 14, 14]  x[2] [B,72,16,16]  1x1 conv(72→128) + bilinear→14
        # [B, 512,  7,  7]  x[3] [B,144,8,8]   1x1 conv(144→512)+ bilinear→7
        #
        # 选择 x[1]/x[2]/x[3] 而非 x[0]：
        #   - x[0] 分辨率最高(64×64)但通道最少(18)，语义最弱，不适合做 query
        #   - x[1~3] 分辨率递减、通道递增，与 MobileFaceNet out3/out4/conv 的
        #     语义层次（浅→深）完全对应

        self.adapter1 = nn.Sequential(
            nn.Conv2d(hrnet_channels[1], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.adapter2 = nn.Sequential(
            nn.Conv2d(hrnet_channels[2], 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.adapter3 = nn.Sequential(
            nn.Conv2d(hrnet_channels[3], 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Adapter 权重可学习（即使 HRNet 冻结），让 POSTER++ 可以端到端微调
        print('[HRNetLandmarkBackbone] Adapters initialized (trainable).')

    # ── 前向传播 ───────────────────────────────────────────────────────
    def forward(self, x):
        """
        x: [B, 3, 112, 112]

        Returns:
            feat1: [B,  64, 28, 28]   对应 MobileFaceNet out3
            feat2: [B, 128, 14, 14]   对应 MobileFaceNet out4
            feat3: [B, 512,  7,  7]   对应 MobileFaceNet conv_features
        """
        # ── HRNet 前向（截止到 stage4，跳过 head/heatmap）──
        hrnet_feats = self._hrnet_features(x)   # list of 4 tensors

        # ── Adapter + resize ──
        feat1 = F.interpolate(self.adapter1(hrnet_feats[1]),
                              size=(28, 28), mode='bilinear', align_corners=False)
        feat2 = F.interpolate(self.adapter2(hrnet_feats[2]),
                              size=(14, 14), mode='bilinear', align_corners=False)
        feat3 = F.interpolate(self.adapter3(hrnet_feats[3]),
                              size=(7, 7),  mode='bilinear', align_corners=False)

        return feat1, feat2, feat3

    def _hrnet_features(self, x):
        """
        复制 HighResolutionNet.forward() 逻辑，但在 head 之前 return。
        返回 stage4 的四个分支特征图列表。
        """
        net = self.hrnet

        # stem
        x = net.relu(net.bn1(net.conv1(x)))
        x = net.relu(net.bn2(net.conv2(x)))
        x = net.layer1(x)

        # stage2
        x_list = []
        for i in range(net.stage2_cfg['NUM_BRANCHES']):
            if net.transition1[i] is not None:
                x_list.append(net.transition1[i](x))
            else:
                x_list.append(x)
        y_list = net.stage2(x_list)

        # stage3
        x_list = []
        for i in range(net.stage3_cfg['NUM_BRANCHES']):
            if net.transition2[i] is not None:
                x_list.append(net.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = net.stage3(x_list)

        # stage4
        x_list = []
        for i in range(net.stage4_cfg['NUM_BRANCHES']):
            if net.transition3[i] is not None:
                x_list.append(net.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        stage4_out = net.stage4(x_list)   # list of 4 tensors，不经过 head

        return stage4_out


# ─────────────────────────────────────────────────────────────────────────────
# 如何修改 PosterV2_7cls.py
# ─────────────────────────────────────────────────────────────────────────────
PATCH_INSTRUCTIONS = """
修改 models/PosterV2_7cls.py，共 3 处改动：

① 顶部 import（替换 MobileFaceNet import）：
   # 删除：
   from .mobilefacenet import MobileFaceNet
   # 新增：
   from hrnet_landmark_backbone import HRNetLandmarkBackbone

② __init__ 中替换模型实例化（约第 234-244 行）：
   # 删除：
   self.face_landback = MobileFaceNet([112, 112], 136)
   face_landback_checkpoint = torch.load(r'...mobilefacenet_model_best.pth.tar', ...)
   self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])
   for param in self.face_landback.parameters():
       param.requires_grad = False

   # 新增：
   from lib.config import config as hrnet_cfg, update_config
   update_config(hrnet_cfg, type('args', (), {
       'cfg': 'experiments/300w/hrnet-r90jt.yaml',
       'model_file': 'infanface_pretrained/hrnet-r90jt.pth'
   })())
   self.face_landback = HRNetLandmarkBackbone(
       hrnet_config   = hrnet_cfg,
       pretrained_path= 'infanface_pretrained/hrnet-r90jt.pth',
       freeze_hrnet   = True,     # 与原 MobileFaceNet 保持一致
   )

③ forward() 中无需任何修改。
   x_face1, x_face2, x_face3 = self.face_landback(x_face)
   这行代码完全不变，因为输出形状与 MobileFaceNet 完全一致。
"""
