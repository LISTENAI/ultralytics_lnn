#!/usr/bin/env python3
"""
二维码检测模型训练脚本
训练轻量级二维码检测网络 (QRDetector)
"""

import os
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from qrcode_models import QRDetector

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


def export_quant_onnx(model, output_path, input_shape=(1, 3, 320, 320), opset=12):
    """导出量化 ONNX 模型"""
    if not LINGER_AVAILABLE:
        print("⚠️ linger 不可用，无法导出")
        return

    model.eval()
    # 获取模型设备并创建对应设备的dummy input
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


class QRDetectionDataset(Dataset):
    """二维码检测数据集"""
    def __init__(self, data_dir: Path, split: str = 'train', img_size: int = 320,
                 augment: bool = True):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        # 加载图像和标注
        img_dir = data_dir / split / 'images'
        label_dir = data_dir / split / 'labels'

        self.image_files = sorted(list(img_dir.glob('*.jpg')))
        self.label_files = [label_dir / f"{img.stem}.txt" for img in self.image_files]

        print(f"Loaded {len(self.image_files)} {split} samples")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]

        # 数据增强
        if self.augment:
            # 随机水平翻转
            if random.random() < 0.5:
                img = cv2.flip(img, 1)

            # 随机亮度调整
            if random.random() < 0.5:
                brightness = random.uniform(0.8, 1.2)
                img = np.clip(img * brightness, 0, 255).astype(np.uint8)

            # 随机对比度调整
            if random.random() < 0.5:
                contrast = random.uniform(0.8, 1.2)
                img = np.clip((img - 128) * contrast + 128, 0, 255).astype(np.uint8)

        # 缩放到固定尺寸
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 归一化
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        # 创建标签
        prob_label = torch.zeros(1, self.img_size, self.img_size)
        angle_label = torch.tensor(0, dtype=torch.long)

        # 读取标注
        label_path = self.label_files[idx]
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO格式: class x_center y_center width height
                        _, cx, cy, w, h = map(float, parts[:5])

                        # 转换为像素坐标
                        cx_px = int(cx * self.img_size)
                        cy_px = int(cy * self.img_size)
                        bw_px = int(w * self.img_size)
                        bh_px = int(h * self.img_size)

                        # 绘制高斯概率图
                        x1 = max(0, cx_px - bw_px // 2)
                        y1 = max(0, cy_px - bh_px // 2)
                        x2 = min(self.img_size, cx_px + bw_px // 2)
                        y2 = min(self.img_size, cy_px + bh_px // 2)

                        # 使用高斯衰减
                        y_grid, x_grid = np.ogrid[:self.img_size, :self.img_size]
                        dist = np.sqrt((x_grid - cx_px)**2 + (y_grid - cy_px)**2)
                        sigma = max(bw_px, bh_px) // 4
                        gaussian = np.exp(-dist**2 / (2 * sigma**2 + 1e-6))

                        prob_label[0] = np.maximum(prob_label[0], gaussian)

        return img, prob_label, angle_label


class DiceLoss(nn.Module):
    """Dice Loss for detection"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """BCE + Dice Loss for detection"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class AngleLoss(nn.Module):
    """角度分类损失"""
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_det_loss = 0
    total_angle_loss = 0

    criterion_det = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    criterion_angle = AngleLoss()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, prob_labels, angle_labels) in enumerate(pbar):
        images = images.to(device)
        prob_labels = prob_labels.to(device)
        angle_labels = angle_labels.to(device)

        # 前向传播
        outputs = model(images)

        # 计算损失
        det_loss = criterion_det(outputs['prob_map'], prob_labels)
        angle_loss = criterion_angle(outputs['angle_logits'], angle_labels)

        # 总损失
        loss = det_loss + 0.5 * angle_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_det_loss += det_loss.item()
        total_angle_loss += angle_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'det': f'{det_loss.item():.4f}',
            'angle': f'{angle_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_det_loss = total_det_loss / len(dataloader)
    avg_angle_loss = total_angle_loss / len(dataloader)

    return {
        'loss': avg_loss,
        'det_loss': avg_det_loss,
        'angle_loss': avg_angle_loss
    }


def validate(model, dataloader, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_det_loss = 0

    criterion_det = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    with torch.no_grad():
        for images, prob_labels, angle_labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            prob_labels = prob_labels.to(device)

            outputs = model(images)
            det_loss = criterion_det(outputs['prob_map'], prob_labels)

            total_loss += det_loss.item()
            total_det_loss += det_loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"   Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train QR Code Detector')
    parser.add_argument('--data-dir', type=str,
                       default='dataset/qrcode_detection',
                       help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=320, help='Image size')
    parser.add_argument('--output', type=str, default='runs/qrcode_detect',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    # 量化训练参数
    parser.add_argument('--quant', action='store_true', help='Enable quantization training')
    parser.add_argument('--quant-stage', type=str, default='both',
                       choices=['both', 'constrain', 'quant'],
                       help='Quantization stage: both(constrain+quant), constrain only, quant only')
    parser.add_argument('--quant-config', type=str, default=None,
                       help='Quantization config file path')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    data_dir = Path(args.data_dir)
    train_dataset = QRDetectionDataset(data_dir, 'train', args.img_size)
    val_dataset = QRDetectionDataset(data_dir, 'val', args.img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=2, pin_memory=True)

    # 模型
    model = QRDetector(in_channels=3, num_angle_classes=4)
    model = model.to(device)

    # 量化训练处理
    quant_config_file = args.quant_config
    if quant_config_file is None:
        # 使用默认配置
        quant_config_file = str(TASK_DIR / 'cfg' / 'quant_config.yaml')

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # 优化器 (用于量化训练和普通训练)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 最佳损失初始化
    best_loss = float('inf')

    if args.quant:
        print(f"\n🔧 量化训练模式: {args.quant_stage}")
        if args.quant_stage in ('both', 'constrain'):
            # 阶段1: 约束训练
            constrain_epochs = args.epochs // 2 if args.quant_stage == 'both' else args.epochs
            print(f"   阶段1: 约束训练 ({constrain_epochs} epochs)")
            model = add_quantization(model, quant_config_file, 'constrain')

            # 约束训练循环
            for epoch in range(1, constrain_epochs + 1):
                train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
                val_loss = validate(model, val_loader, device)
                scheduler.step()

                print(f"   Epoch {epoch}/{constrain_epochs}")
                print(f"     Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_path = output_dir / 'best_constrain.pth'
                    save_checkpoint(model, optimizer, epoch, val_loss, save_path)

            # 保存约束模型
            print(f"   约束训练完成，保存模型")

        if args.quant_stage in ('both', 'quant'):
            # 阶段2: 量化训练
            if args.quant_stage == 'both':
                # 加载最佳约束模型 (如果存在)
                checkpoint_path = output_dir / 'best_constrain.pth'
                load_success = False
                if checkpoint_path.exists():
                    print(f"   尝试加载约束模型: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        print("   约束模型加载成功")
                        load_success = True
                    except Exception as e:
                        print(f"   约束模型加载失败: {e}")
                        # 重新创建浮点模型
                        model = QRDetector(in_channels=3, num_angle_classes=4)
                        model = model.to(device)
                else:
                    print("   未找到约束模型，使用浮点模型")

                # 如果约束模型加载成功，linger 模块可能导致量化失败
                # 这种情况使用浮点模型继续
                if load_success:
                    print("   警告: 约束模型包含 linger 模块，可能导致量化失败")
                    print("   重新创建浮点模型用于量化训练")
                    model = QRDetector(in_channels=3, num_angle_classes=4)
                    model = model.to(device)

                # 切换到量化模式
                print(f"   阶段2: 量化训练 ({args.epochs // 2} epochs)")
                try:
                    model = add_quantization(model, quant_config_file, 'quant')
                except Exception as e:
                    print(f"   ⚠️ 量化训练失败: {e}")
                    print("   跳过量化训练")
                    return

                quant_epochs = args.epochs // 2
            else:
                print(f"   量化训练 ({args.epochs} epochs)")
                try:
                    model = add_quantization(model, quant_config_file, 'quant')
                except Exception as e:
                    print(f"   ⚠️ 量化训练失败: {e}")
                    print("   跳过量化训练")
                    return
                quant_epochs = args.epochs

            # 确保模型完全在 CUDA 上
            model = model.to(device)
            model.train()

            # 量化训练循环 (使用更小的学习率)
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=quant_epochs)

            for epoch in range(1, quant_epochs + 1):
                train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
                val_loss = validate(model, val_loader, device)
                scheduler.step()

                print(f"   Epoch {epoch}/{quant_epochs}")
                print(f"     Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_path = output_dir / 'best_detector.pth'
                    save_checkpoint(model, optimizer, epoch, val_loss, save_path)

            # 导出量化 ONNX
            onnx_path = output_dir / 'best_detector.onnx'
            export_quant_onnx(model, str(onnx_path))

            print("\n✅ 量化训练完成!")
            print(f"   模型: {output_dir / 'best_detector.pth'}")
            print(f"   ONNX: {onnx_path}")
            return

    # 训练
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50)

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # 验证
        val_loss = validate(model, val_loader, device)

        # 更新学习率
        scheduler.step()

        # 打印信息
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (det: {train_metrics['det_loss']:.4f}, angle: {train_metrics['angle_loss']:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = output_dir / 'best_detector.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)

        # 定期保存
        if epoch % 10 == 0:
            save_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)

    print("\n✅ 训练完成!")
    print(f"   模型保存路径: {output_dir / 'best_detector.pth'}")


if __name__ == "__main__":
    main()