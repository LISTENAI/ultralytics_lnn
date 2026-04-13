#!/usr/bin/env python3
"""
自动驾驶玩具车 - 道路分割训练脚本
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

# ==================== linger 量化训练支持 ====================
# 添加 linger 路径
SCRIPT_DIR = Path(__file__).parent.resolve()
TASK_DIR = SCRIPT_DIR.parent
ULTRALYTICS_PATH = TASK_DIR.parent.parent
LINGER_PATH = ULTRALYTICS_PATH / 'thirdparty' / 'linger'
if LINGER_PATH.exists() and str(LINGER_PATH) not in sys.path:
    sys.path.insert(0, str(LINGER_PATH))

# 清理 linger 模块缓存
if 'linger' in sys.modules:
    del sys.modules['linger']
for mod in list(sys.modules.keys()):
    if mod.startswith('linger'):
        del sys.modules[mod]

# linger 可用性检查
LINGER_AVAILABLE = False
try:
    import linger
    from linger import init, constrain
    LINGER_AVAILABLE = True
except ImportError:
    pass


def add_quantization(model, config_file=None, stage='quant'):
    """在模型中添加量化"""
    if not LINGER_AVAILABLE:
        print("⚠️ linger 不可用，跳过量量化")
        return model

    if stage == 'constrain':
        print(f"🔧 添加浮点约束训练...")
        return constrain(model, config_file=config_file)
    else:
        print(f"🔧 添加量化训练...")
        return init(model, config_file=config_file)


def export_quant_onnx(model, output_path, input_shape=(1, 3, 160, 160), opset=12):
    """导出量化 ONNX 模型"""
    if not LINGER_AVAILABLE:
        print("⚠️ linger 不可用，无法导出")
        return

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        linger.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    print(f"✅ 量化模型已导出: {output_path}")

# 添加脚本路径
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from road_segnet import build_road_segnet
from prepare_dataset import RoadDataset


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """组合损失：CrossEntropy + Dice + 决策损失"""
    def __init__(self, seg_weight=1.0, dice_weight=0.5, decision_weight=0.3):
        super().__init__()
        self.seg_weight = seg_weight
        self.dice_weight = dice_weight
        self.decision_weight = decision_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, seg_logit, decision_logit, target_mask, target_decision):
        # 分割损失
        seg_loss = self.ce(seg_logit, target_mask) + self.dice(seg_logit, target_mask)

        # 决策损失（如果有决策标签）
        if target_decision is not None:
            decision_loss = F.cross_entropy(decision_logit, target_decision)
        else:
            # 从分割结果推断决策（简化版）
            # 根据道路中心位置判断方向
            decision_loss = torch.tensor(0.0, device=seg_logit.device)

        total_loss = self.seg_weight * seg_loss + self.decision_weight * decision_loss

        return total_loss, seg_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_seg_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # 前向传播
        seg_logit, decision = model(images)

        # 计算损失（无决策标签，使用None）
        loss, seg_loss = criterion(seg_logit, decision, masks, None)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'seg': f'{seg_loss.item():.4f}'
        })

    return total_loss / len(dataloader), total_seg_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            seg_logit, decision = model(images)
            loss, _ = criterion(seg_logit, decision, masks, None)

            total_loss += loss.item()

            # 计算IoU
            pred = seg_logit.argmax(dim=1)
            iou = compute_iou(pred, masks)
            total_iou += iou

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def compute_iou(pred, target, num_classes=3):
    """计算IoU"""
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(0.0)
    return np.mean(ious)


def generate_decision_from_segmentation(masks):
    """
    从分割结果生成决策标签
    用于自监督学习
    """
    batch_size = masks.size(0)
    decisions = torch.zeros(batch_size, dtype=torch.long, device=masks.device)

    for i in range(batch_size):
        mask = masks[i].cpu().numpy()
        h, w = mask.shape

        # 计算道路左右两侧的比例
        left_ratio = (mask[:, :w//3] == 1).sum() / (h * w // 3)  # 左侧道路
        center_ratio = (mask[:, w//3:2*w//3] == 1).sum() / (h * w // 3)  # 中间道路
        right_ratio = (mask[:, 2*w//3:] == 1).sum() / (h * w // 3)  # 右侧道路

        # 决策逻辑
        if center_ratio > 0.5:
            decisions[i] = 0  # 直行
        elif left_ratio > right_ratio * 1.5:
            decisions[i] = 1  # 左转
        elif right_ratio > left_ratio * 1.5:
            decisions[i] = 2  # 右转
        else:
            decisions[i] = 3  # 停止（无明显道路）

    return decisions


def main():
    parser = argparse.ArgumentParser(description='自动驾驶玩具车训练')
    parser.add_argument('--data-dir', type=str,
                       default='dataset/toy_car_road',
                       help='数据集目录')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--input-size', type=int, default=160,
                       help='输入尺寸')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--output', type=str,
                       default='runs/toy_car_autodrive',
                       help='输出目录')

    # 量化训练参数
    parser.add_argument('--quant', action='store_true', help='Enable quantization training')
    parser.add_argument('--quant-stage', type=str, default='both',
                       choices=['both', 'constrain', 'quant'],
                       help='Quantization stage')
    parser.add_argument('--quant-config', type=str, default=None,
                       help='Quantization config file path')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = RoadDataset(args.data_dir, split='train', input_size=args.input_size)
    if len(train_dataset) == 0:
        # 尝试 val split
        train_dataset = RoadDataset(args.data_dir, split='val', input_size=args.input_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"训练样本数: {len(train_dataset)}")

    # 创建模型
    print("\n创建模型...")
    model = build_road_segnet(num_classes=3, input_size=args.input_size)
    model = model.to(device)

    params = model.count_parameters()
    print(f"模型参数量: {params:,} ({params/1e6:.2f}M)")
    if params > 4e6:
        print("⚠️ 警告: 参数量超过4M!")

    # 损失函数和优化器
    criterion = CombinedLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 量化配置
    quant_config_file = args.quant_config
    if quant_config_file is None:
        quant_config_file = str(TASK_DIR / 'cfg' / 'quant_config.yaml')

    # 量化训练处理
    if hasattr(args, 'quant') and args.quant:
        print(f"\n🔧 量化训练模式: {args.quant_stage}")
        if args.quant_stage in ('both', 'constrain'):
            constrain_epochs = args.epochs // 2 if args.quant_stage == 'both' else args.epochs
            print(f"   阶段1: 约束训练 ({constrain_epochs} epochs)")
            model = add_quantization(model, quant_config_file, 'constrain')

            for epoch in range(1, constrain_epochs + 1):
                train_loss, train_seg_loss = train_epoch(
                    model, train_loader, optimizer, criterion, device, epoch
                )
                scheduler.step()
                print(f"   Epoch {epoch}/{constrain_epochs}: loss={train_loss:.4f}")

            torch.save(model.state_dict(), output_dir / 'best_constrain.pth')
            print("   约束训练完成")

        if args.quant_stage in ('both', 'quant'):
            print(f"   阶段2: 量化训练")
            try:
                model = add_quantization(model, quant_config_file, 'quant')
            except Exception as e:
                print(f"   ⚠️ 量化训练失败: {e}")
                return

            model = model.to(device)
            model.train()
            for param in model.parameters():
                param.requires_grad = True

            optimizer = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs // 2, eta_min=1e-6)

            quant_epochs = args.epochs // 2 if args.quant_stage == 'both' else args.epochs
            for epoch in range(1, quant_epochs + 1):
                train_loss, train_seg_loss = train_epoch(
                    model, train_loader, optimizer, criterion, device, epoch
                )
                scheduler.step()
                print(f"   Epoch {epoch}/{quant_epochs}: loss={train_loss:.4f}")

            torch.save(model.state_dict(), output_dir / 'best_detector.pth')

            # 导出 ONNX
            onnx_path = output_dir / 'best_detector.onnx'
            export_quant_onnx(model, str(onnx_path), input_shape=(1, 3, args.input_size, args.input_size))
            print("\n✅ 量化训练完成!")
            return

    # ���练循环
    print("\n开始训练...")
    best_iou = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_seg_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        scheduler.step()

        # 每5个epoch验证一次
        if epoch % 5 == 0:
            val_loss, val_iou = validate(model, train_loader, criterion, device)
            print(f"Epoch {epoch}: loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_iou={val_iou:.4f}")

            # 保存最佳模型
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                }, output_dir / 'best_model.pth')
                print(f"✓ 保存最佳模型 (IoU: {best_iou:.4f})")

    print(f"\n训练完成! 最佳IoU: {best_iou:.4f}")
    print(f"模型保存位置: {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()