#!/usr/bin/env python3
"""
使用颜色分割方法为BDD100K生成伪标签
基于道路颜色特征进行简单的分割
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def color_based_segmentation(image_path):
    """
    基于颜色特征的简单道路分割
    道路通常是灰色/暗色调，靠近图像底部
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 转换为HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 道路颜色范围（灰色、暗色）
    # 饱和度较低，明度较低
    lower_gray = np.array([0, 0, 30])
    upper_gray = np.array([180, 50, 120])

    # 提取灰色区域（可能是道路）
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # 只考虑图像下半部分（道路通常在下方）
    bottom_half = gray_mask[h//2:, :]

    # 找到最大的连通区域
    if bottom_half.sum() > 0:
        # 扩展到上半部分的区域（假设道路向上延伸）
        # 使用漫水填充来扩展
        mask[h//2:, :][bottom_half > 0] = 1

        # 也考虑上半部分的相似颜色区域
        upper_gray2 = cv2.inRange(hsv[:h//2, :], lower_gray, upper_gray)
        # 只保留与下半部分相连的区域
        mask[:h//2, :][upper_gray2 > 0] = 1

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def generate_pseudo_labels(images_dir, output_dir, max_samples=5000):
    """生成伪标签"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 获取图像文件
    image_files = sorted(list(images_dir.glob('*.jpg')))[:max_samples]

    print(f"生成伪标签: {len(image_files)} 图像")

    for img_path in tqdm(image_files):
        mask = color_based_segmentation(img_path)

        if mask is not None:
            # 调整大小为160x160
            mask_resized = cv2.resize(mask, (160, 160), interpolation=cv2.INTER_NEAREST)

            # 保存掩码
            mask_path = masks_dir / (img_path.stem + '.png')
            Image.fromarray(mask_resized).save(mask_path)

    print(f"伪标签已保存: {masks_dir}")
    return masks_dir


def train_with_pseudo_labels(model_path, images_dir, masks_dir, epochs=30, args=None):
    batch_size = args.batch_size if args else 16
    """使用伪标签训练模型"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm

    sys.path.insert(0, str(Path(__file__).parent))
    from road_segnet import build_road_segnet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集类
    class PseudoLabelDataset(Dataset):
        def __init__(self, img_dir, mask_dir):
            self.img_dir = Path(img_dir)
            self.mask_dir = Path(mask_dir)

            # 获取有效样本
            self.samples = []
            for mf in self.mask_dir.glob('*.png'):
                img_path = self.img_dir / (mf.stem + '.jpg')
                if img_path.exists():
                    self.samples.append((img_path, mf))

            print(f"有效样本: {len(self.samples)}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, mask_path = self.samples[idx]

            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # 随机翻转增强
            if np.random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            image = image.resize((160, 160), Image.BILINEAR)
            mask = mask.resize((160, 160), Image.NEAREST)

            image_np = np.array(image)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(np.array(mask)).long()

            # 将0值（背景）改为2（障碍物），保留1（道路）
            # 简化：只区分道路和非道路
            simple_mask = torch.zeros_like(mask_tensor)
            simple_mask[mask_tensor == 1] = 1  # 道路
            simple_mask[mask_tensor == 0] = 0  # 背景/障碍物

            return image_tensor, simple_mask

    # 准备数据集
    dataset = PseudoLabelDataset(images_dir, masks_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 加载模型
    model = build_road_segnet(num_classes=2, input_size=160)
    if args.model and Path(args.model).exists():
        try:
            checkpoint = torch.load(args.model, map_location=device, weights_only=False)
            # 尝试加载，跳过不匹配的层
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model_state_dict']
            # 过滤掉不匹配的层
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print("加载了部分预训练权重")
        except Exception as e:
            print(f"无法加载预训练模型: {e}")

    model = model.to(device)

    params = model.count_parameters()
    print(f"模型参数量: {params:,} ({params/1e6:.2f}M)")

    # 训练
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n开始训练 {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            seg_logit, _ = model(images)
            loss = criterion(seg_logit, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

        # 每10个epoch保存一次
        if epoch % 10 == 0:
            output_path = Path('runs/toy_car_bdd') / f'model_{epoch}.pth'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
            }, output_path)
            print(f"  模型已保存: {output_path}")

    # 保存最终模型
    final_path = Path('runs/toy_car_bdd') / 'best_model.pth'
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
    }, final_path)
    print(f"\n模型已保存: {final_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                       default='/CodeRepo/Code/dwwang16/dataset/toy_car_road',
                       help='数据根目录')
    parser.add_argument('--model', type=str, default='runs/toy_car/best_model.pth',
                       help='预训练模型')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='最大样本数')
    args = parser.parse_args()

    images_dir = Path(args.data_dir) / "100k" / "train"
    output_dir = Path(args.data_dir) / "pseudo_labels"

    # 步骤1: 生成伪标签
    print("="*60)
    print("步骤1: 生成伪标签")
    print("="*60)
    masks_dir = generate_pseudo_labels(images_dir, output_dir, args.max_samples)

    # 步骤2: 训练模型
    print("\n" + "="*60)
    print("步骤2: 训练模型")
    print("="*60)
    train_with_pseudo_labels(args.model, images_dir, masks_dir, args.epochs, args)


if __name__ == '__main__':
    main()