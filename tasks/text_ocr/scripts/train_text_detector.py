#!/usr/bin/env python3
"""
文本检测模型训练脚本
训练轻量级文本检测网络 (TextDetector)
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


def export_quant_onnx(model, output_path, input_shape=(1, 3, 640, 640), opset=12):
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

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from text_models import TextDetector


class TextDetectionDataset(Dataset):
    """文本检测数据集"""
    def __init__(self, data_dir: Path, split: str = 'train', img_size: int = 640, val_ratio: float = 0.1):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size

        # 加载图像和标注 - 尝试多个可能的目录结构
        # 1. data_dir/train/images
        # 2. data_dir/images (扁平结构)
        img_dir = data_dir / split / 'images'
        if not img_dir.exists():
            # 尝试使用 data_dir/images
            img_dir = data_dir / 'images'

        label_dir = data_dir / split / 'labels'
        if not label_dir.exists():
            # 尝试使用 data_dir/annotations
            label_dir = data_dir / 'annotations'

        self.image_files = []
        if img_dir.exists():
            # 支持多种图像格式
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                self.image_files.extend(sorted(list(img_dir.glob(ext))))

        # 如果没有找到图像，尝试从 data_dir/images 加载并划分
        if len(self.image_files) == 0:
            img_dir = data_dir / 'images'
            if img_dir.exists():
                all_images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    all_images.extend(sorted(list(img_dir.glob(ext))))

                # 划分训练集和验证集
                import random
                random.seed(42)
                random.shuffle(all_images)
                val_size = int(len(all_images) * val_ratio)
                if split == 'train':
                    self.image_files = all_images[val_size:]
                else:
                    self.image_files = all_images[:val_size]
                print(f"  Split from {len(all_images)} images: {split}={len(self.image_files)}")

        self.label_files = []
        for img in self.image_files:
            # 尝试多种标签文件位置
            label_name = img.stem + '.txt'
            possible_paths = [
                data_dir / 'annotations' / label_name,
                data_dir / split / 'labels' / label_name,
                label_dir / label_name,
            ]
            label_path = None
            for p in possible_paths:
                if p.exists():
                    label_path = p
                    break
            self.label_files.append(label_path)

        print(f"Loaded {len(self.image_files)} {split} samples from {img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]

        # 缩放到固定尺寸
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 归一化
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        # 创建标签 (文本概率图)
        label = torch.zeros(1, self.img_size, self.img_size)

        # 读取标注
        label_path = self.label_files[idx]
        if label_path is not None and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO格式: class x_center y_center width height
                        _, cx, cy, w, h = map(float, parts[:5])

                        # 转换为像素坐标
                        cx = int(cx * self.img_size)
                        cy = int(cy * self.img_size)
                        bw = int(w * self.img_size)
                        bh = int(h * self.img_size)

                        # 绘制高斯概率图
                        x1 = max(0, cx - bw // 2)
                        y1 = max(0, cy - bh // 2)
                        x2 = min(self.img_size, cx + bw // 2)
                        y2 = min(self.img_size, cy + bh // 2)

                        # 使用高斯衰减
                        y_grid, x_grid = np.ogrid[:self.img_size, :self.img_size]
                        center_x, center_y = cx, cy
                        dist = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                        gaussian = np.exp(-dist**2 / (2 * (max(bw, bh) // 4)**2))

                        label[0] = np.maximum(label[0], gaussian)

        return img, label


class DiceLoss(nn.Module):
    """Dice Loss for text detection"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred_prob, target):
        bce_loss = self.bce(pred_prob, target)
        dice_loss = self.dice(pred_prob, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(imgs)
        pred_prob = outputs['prob_map']

        # 计算损失
        loss = criterion(pred_prob, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='Validating'):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            pred_prob = outputs['prob_map']

            loss = criterion(pred_prob, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main(args):
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    data_dir = Path(args.data_dir)
    train_dataset = TextDetectionDataset(data_dir, 'train', args.img_size)
    val_dataset = TextDetectionDataset(data_dir, 'val', args.img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # 模型
    model = TextDetector().to(device)
    params = model.count_parameters()
    print(f"Model parameters: {params:,} ({params/1e6:.2f}M)")

    if params > 4_000_000:
        print("⚠️ Warning: Model parameters exceed 4M!")

    # 损失函数和优化器
    criterion = CombinedLoss(alpha=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 最佳损失初始化
    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 量化配置
    quant_config_file = args.quant_config
    if quant_config_file is None:
        quant_config_file = str(Path(__file__).parent.parent / 'cfg' / 'quant_config.yaml')

    # 量化训练处理
    if hasattr(args, 'quant') and args.quant:
        print(f"\n🔧 量化训练模式: {args.quant_stage}")
        if args.quant_stage in ('both', 'constrain'):
            # 阶段1: 约束训练
            constrain_epochs = args.epochs // 2 if args.quant_stage == 'both' else args.epochs
            print(f"   阶段1: 约束训练 ({constrain_epochs} epochs)")
            model = add_quantization(model, quant_config_file, 'constrain')

            for epoch in range(1, constrain_epochs + 1):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
                val_loss = validate(model, val_loader, criterion, device)
                scheduler.step()
                print(f"   Epoch {epoch}/{constrain_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss},
                              output_dir / 'best_constrain.pth')

        if args.quant_stage in ('both', 'quant'):
            # 阶段2: 量化训练
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

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs // 2)

            quant_epochs = args.epochs // 2 if args.quant_stage == 'both' else args.epochs
            for epoch in range(1, quant_epochs + 1):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
                val_loss = validate(model, val_loader, criterion, device)
                scheduler.step()
                print(f"   Epoch {epoch}/{quant_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss},
                              output_dir / 'best_detector.pth')

            # 导出 ONNX
            onnx_path = output_dir / 'best_detector.onnx'
            export_quant_onnx(model, str(onnx_path), input_shape=(1, 3, args.img_size, args.img_size))
            print("\n✅ 量化训练完成!")
            return

    # 普通训练
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_detector.pth')
            print(f"   ✅ Best model saved (val_loss={val_loss:.4f})")

        # 定期保存
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / f'detector_epoch_{epoch}.pth')

    print(f"\n✅ Training complete! Best val_loss: {best_loss:.4f}")
    print(f"   Model saved to: {output_dir / 'best_detector.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Text Detector')
    parser.add_argument('--data-dir', type=str, default='/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/dataset/text_detection',
                        help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='runs/text_ocr/detector',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save-freq', type=int, default=10, help='Save frequency')

    # 量化训练参数
    parser.add_argument('--quant', action='store_true', help='Enable quantization training')
    parser.add_argument('--quant-stage', type=str, default='both',
                       choices=['both', 'constrain', 'quant'],
                       help='Quantization stage')
    parser.add_argument('--quant-config', type=str, default=None,
                       help='Quantization config file path')

    args = parser.parse_args()
    main(args)