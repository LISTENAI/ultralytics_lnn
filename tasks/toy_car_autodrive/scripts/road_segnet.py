"""
轻量级道路分割模型
专为自动驾驶玩具车设计，参数 < 4M

架构设计：
- 编码器：MobileNetV2 风格深度可分离卷积
- 解码器：轻量级上采样 + 特征融合
- 输出：道路分割掩码 + 决策类别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 减少参数量"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差结构"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio
        layers = []

        if expand_ratio != 1:
            # 扩展阶段
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # 深度可分离卷积
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 投影
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EncoderBlock(nn.Module):
    """编码器块"""
    def __init__(self, in_channels, out_channels, num_blocks, stride=2):
        super().__init__()
        layers = []

        # 第一个块可能改变尺寸
        layers.append(InvertedResidual(in_channels, out_channels, stride))

        # 后续块保持尺寸
        for _ in range(1, num_blocks):
            layers.append(InvertedResidual(out_channels, out_channels, 1))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class LightDecoder(nn.Module):
    """轻量级解码器"""
    def __init__(self, encoder_channels, decoder_channels, num_classes):
        super().__init__()

        # 逐通道上采样
        self.up1 = nn.ConvTranspose2d(encoder_channels[3], decoder_channels[2], 4, 2, 1)
        self.conv1 = DepthwiseSeparableConv(decoder_channels[2] + encoder_channels[2], decoder_channels[1])

        self.up2 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[1], 4, 2, 1)
        self.conv2 = DepthwiseSeparableConv(decoder_channels[1] + encoder_channels[1], decoder_channels[0])

        self.up3 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[0], 4, 2, 1)
        self.conv3 = DepthwiseSeparableConv(decoder_channels[0] + encoder_channels[0], 64)

        # 最终上采样到原始分辨率
        self.final_up = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv_final = nn.Conv2d(32, num_classes, 1)

    def forward(self, skip1, skip2, skip3, feat4):
        # feat4: 最深特征
        x = self.up1(feat4)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv3(x)

        x = self.final_up(x)
        x = self.conv_final(x)
        return x


class DecisionHead(nn.Module):
    """决策头 - 输出转向决策"""
    def __init__(self, feature_channels, num_decisions=4):
        """
        决策类别:
        0: 直行
        1: 左转
        2: 右转
        3: 停止
        """
        super().__init__()
        # 全局平均池化 + 全连接
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_decisions)
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RoadSegNet(nn.Module):
    """
    道路分割网络 - 专为玩具车设计
    参数量: ~2.1M
    """
    def __init__(self, num_classes=2, input_size=160):
        super().__init__()
        self.num_classes = num_classes

        # 输入 stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # 160->80
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # 编码器阶段
        # channels: [32, 96, 144, 192]
        self.encoder1 = EncoderBlock(32, 32, 1, stride=1)      # 80->80
        self.encoder2 = EncoderBlock(32, 64, 2, stride=2)      # 80->40
        self.encoder3 = EncoderBlock(64, 96, 3, stride=2)     # 40->20
        self.encoder4 = EncoderBlock(96, 128, 2, stride=2)    # 20->10

        # 解码器channels
        decoder_ch = [96, 64, 32]

        # 解码器
        self.decoder = LightDecoder(
            [32, 64, 96, 128],  # encoder channels
            decoder_ch,
            num_classes
        )

        # 决策头
        self.decision = DecisionHead(128, num_decisions=4)

    def forward(self, x):
        # Stem
        x0 = self.stem(x)  # B, 32, H/2, W/2

        # Encoder
        e1 = self.encoder1(x0)   # B, 32, H/2, W/2
        e2 = self.encoder2(e1)    # B, 64, H/4, W/4
        e3 = self.encoder3(e2)    # B, 96, H/8, W/8
        e4 = self.encoder4(e3)   # B, 128, H/16, W/16

        # Decoder
        seg_logit = self.decoder(e1, e2, e3, e4)  # B, num_classes, H, W

        # Decision
        decision = self.decision(e4)  # B, 4

        return seg_logit, decision

    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_road_segnet(num_classes=2, input_size=160):
    """构建道路分割网络"""
    model = RoadSegNet(num_classes=num_classes, input_size=input_size)
    return model


# 测试
if __name__ == '__main__':
    model = build_road_segnet(num_classes=2, input_size=160)
    params = model.count_parameters()
    print(f"模型参数量: {params:,} ({params/1e6:.2f}M)")

    # 测试前向传播
    x = torch.randn(1, 3, 160, 160)
    seg, dec = model(x)
    print(f"分割输出形状: {seg.shape}")  # (1, 2, 160, 160)
    print(f"决策输出形状: {dec.shape}")  # (1, 4)
    print(f"决策类别: {torch.softmax(dec, dim=1)}")