#!/usr/bin/env python3
"""
使用预训练分割模型生成伪标签
然后用伪标签训练轻量级道路分割模型
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 添加路径
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from road_segnet import build_road_segnet


# BDD100K 类别到简单类别的映射
# 道路类别: 0 (road)
# 障碍物: car, truck, bus, train, motorcycle, bicycle, person, rider, pole, traffic light, traffic sign, building, fence, wall, vegetation
# 背景: sky, terrain
BDD_TO_SIMPLE = {
    0: 1,   # road -> 道路
    1: 2,   # sidewalk -> 障碍物
    2: 2,   # building -> 障碍物
    3: 2,   # wall -> 障碍物
    4: 2,   # fence -> 障碍物
    5: 2,   # pole -> 障碍物
    6: 2,   # traffic light -> 障碍物
    7: 2,   # traffic sign -> 障碍物
    8: 2,   # vegetation -> 障碍物
    9: 0,   # terrain -> 背景
    10: 0,  # sky -> 背景
    11: 2,  # person -> 障碍物
    12: 2,  # rider -> 障碍物
    13: 2,  # car -> 障碍物
    14: 2,  # truck -> 障碍物
    15: 2,  # bus -> 障碍物
    16: 2,  # train -> 障碍物
    17: 2,  # motorcycle -> 障碍物
    18: 2,  # bicycle -> 障碍物
}


class BDD100KImageDataset(Dataset):
    """BDD100K 图像数据集（无标签）"""
    def __init__(self, images_dir, input_size=160):
        self.images_dir = Path(images_dir)
        self.input_size = input_size

        # 获取所有图像
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))[:10000]  # 使用前10000张

        print(f"找到 {len(self.image_files)} 张图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        # 调整大小
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # 转换为tensor
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, str(img_path), original_size


def create_pseudo_labels(model, dataloader, device, output_dir):
    """使用预训练模型生成伪标签"""
    output_dir = Path(output_dir)
    masks_dir = output_dir / "pseudo_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("生成伪标签...")

    model.eval()
    with torch.no_grad():
        for images, paths, sizes in tqdm(dataloader):
            images = images.to(device)

            # 推理
            seg_logit, _ = model(images)
            seg_pred = seg_logit.argmax(dim=1)  # B, H, W

            # 保存伪标签
            for i, (pred, path) in enumerate(zip(seg_pred, paths)):
                path = Path(path)
                mask_np = pred.cpu().numpy().astype(np.uint8)

                # 调整回原始尺寸并保存（可选）

                # 保存为PNG
                mask_path = masks_dir / (path.stem + '.png')
                Image.fromarray(mask_path).save(mask_path)

    print(f"伪标签已保存: {masks_dir}")
    return masks_dir


def train_with_pseudo_labels(model, images_dir, masks_dir, device, epochs=30):
    """使用伪标签训练模型"""
    # 准备数据集
    image_files = sorted(list(Path(images_dir).glob('*.jpg')))[:5000]
    mask_files = {}
    for mf in Path(masks_dir).glob('*.png'):
        mask_files[mf.stem] = mf

    # 过滤有掩码的图像
    valid_pairs = []
    for img_path in image_files:
        if img_path.stem in mask_files:
            valid_pairs.append((img_path, mask_files[img_path.stem]))

    print(f"有效训练样本: {len(valid_pairs)}")

    # 数据集类
    class PseudoLabelDataset(Dataset):
        def __init__(self, pairs, input_size=160):
            self.pairs = pairs
            self.input_size = input_size

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            img_path, mask_path = self.pairs[idx]

            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
            mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)

            image_np = np.array(image)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(np.array(mask)).long()

            return image_tensor, mask_tensor

    dataset = PseudoLabelDataset(valid_pairs)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

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

    return model


def main():
    parser = argparse.ArgumentParser(description='使用伪标签训练')
    parser.add_argument('--data-dir', type=str,
                       default='/CodeRepo/Code/dwwang16/dataset/toy_car_road',
                       help='数据目录')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--output', type=str,
                       default='runs/toy_car_bdd100k',
                       help='输出目录')
    parser.add_argument('--input-size', type=int, default=160,
                       help='输入尺寸')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 检查输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 图像目录
    images_dir = Path(args.data_dir) / "100k" / "train"
    masks_dir = output_dir / "pseudo_masks"

    if not masks_dir.exists() or len(list(masks_dir.glob('*.png'))) < 100:
        # 需要生成伪标签
        print("步骤1: 生成伪标签")

        # 检查是否有预训练分割模型
        try:
            from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

            print("加载预训练 DeepLabV3...")
            pretrained_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            pretrained_model = pretrained_model.to(device)
            pretrained_model.eval()

            # 创建数据集
            dataset = BDD100KImageDataset(images_dir, args.input_size)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            # 生成伪标签
            masks_dir = output_dir / "pseudo_masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            print("使用 DeepLabV3 生成伪标签...")
            with torch.no_grad():
                for images, paths, sizes in tqdm(dataloader):
                    images = images.to(device)

                    # DeepLabV3 推理
                    output = pretrained_model(images)['out']
                    preds = output.argmax(dim=1)

                    # 转换为简单类别
                    simple_preds = torch.zeros_like(preds)
                    for bdd_cls, simple_cls in BDD_TO_SIMPLE.items():
                        simple_preds[preds == bdd_cls] = simple_cls

                    # 保存
                    for i, (pred, path) in enumerate(zip(simple_preds, paths)):
                        path = Path(path)
                        mask_np = pred.cpu().numpy().astype(np.uint8)
                        Image.fromarray(mask_np).save(masks_dir / f"{path.stem}.png")

            print(f"伪标签已保存: {masks_dir}")

        except ImportError as e:
            print(f"无法加载预训练模型: {e}")
            print("使用备选方案：基于图像特征的方法")
            masks_dir = None

    else:
        print(f"使用现有伪标签: {masks_dir}")

    # 步骤2: 训练模型
    print("\n步骤2: 训练模型")

    model = build_road_segnet(num_classes=3, input_size=args.input_size)
    model = model.to(device)

    params = model.count_parameters()
    print(f"模型参数量: {params:,} ({params/1e6:.2f}M)")

    if masks_dir and masks_dir.exists():
        # 使用伪标签训练
        model = train_with_pseudo_labels(
            model, images_dir, masks_dir, device, args.epochs
        )
    else:
        # 使用合成数据训练
        print("使用合成数据集训练...")
        from prepare_dataset import RoadDataset
        dataset = RoadDataset(args.data_dir, split='val', input_size=args.input_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs + 1):
            total_loss = 0
            for images, masks in dataloader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                seg_logit, _ = model(images)
                loss = criterion(seg_logit, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            print(f"Epoch {epoch}/{args.epochs}: loss={total_loss/len(dataloader):.4f}")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': args.epochs,
    }, output_dir / "best_model.pth")

    print(f"\n模型已保存: {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()