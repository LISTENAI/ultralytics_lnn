#!/usr/bin/env python3
"""
文本分割模型训练脚本
训练轻量级文本分割网络 (TextSegmenter)
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

sys.path.insert(0, str(Path(__file__).parent))
from text_models import TextSegmenter


class TextSegmentationDataset(Dataset):
    """文本分割数据集"""
    def __init__(self, data_dir: Path, split: str = 'train', img_size: int = 640):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size

        img_dir = data_dir / split / 'images'
        label_dir = data_dir / split / 'labels'

        self.image_files = sorted(list(img_dir.glob('*.jpg')))
        self.label_files = [label_dir / f"{img.stem}.txt" for img in self.image_files]

        print(f"Loaded {len(self.image_files)} {split} samples")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # 分割标签 (文本区域 + 边缘)
        seg_label = torch.zeros(1, self.img_size, self.img_size)
        edge_label = torch.zeros(1, self.img_size, self.img_size)

        label_path = self.label_files[idx]
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, w, h = map(float, parts[:5])

                        cx = int(cx * self.img_size)
                        cy = int(cy * self.img_size)
                        bw = int(w * self.img_size)
                        bh = int(h * self.img_size)

                        x1 = max(0, cx - bw // 2)
                        y1 = max(0, cy - bh // 2)
                        x2 = min(self.img_size, cx + bw // 2)
                        y2 = min(self.img_size, cy + bh // 2)

                        # 文本区域
                        seg_label[0, y1:y2, x1:x2] = 1.0

                        # 边缘 (宽2像素)
                        edge_thickness = 2
                        edge_label[0, y1:y1+edge_thickness, x1:x2] = 1.0
                        edge_label[0, y2-edge_thickness:y2, x1:x2] = 1.0
                        edge_label[0, y1:y2, x1:x1+edge_thickness] = 1.0
                        edge_label[0, y1:y2, x2-edge_thickness:x2] = 1.0

        return img, seg_label, edge_label


class SegmentationLoss(nn.Module):
    """分割损失函数"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, pred_seg, pred_edge, target_seg, target_edge):
        loss_seg = self.bce(pred_seg, target_seg)
        loss_edge = self.bce(pred_edge, target_edge)
        return loss_seg + self.alpha * loss_edge


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for imgs, seg_labels, edge_labels in pbar:
        imgs = imgs.to(device)
        seg_labels = seg_labels.to(device)
        edge_labels = edge_labels.to(device)

        outputs = model(imgs)
        pred_seg = outputs['seg_mask']
        pred_edge = outputs['edge_mask']

        loss = criterion(pred_seg, pred_edge, seg_labels, edge_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, seg_labels, edge_labels in tqdm(dataloader, desc='Validating'):
            imgs = imgs.to(device)
            seg_labels = seg_labels.to(device)
            edge_labels = edge_labels.to(device)

            outputs = model(imgs)
            pred_seg = outputs['seg_mask']
            pred_edge = outputs['edge_mask']

            loss = criterion(pred_seg, pred_edge, seg_labels, edge_labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_dataset = TextSegmentationDataset(data_dir, 'train', args.img_size)
    val_dataset = TextSegmentationDataset(data_dir, 'val', args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = TextSegmenter().to(device)
    params = model.count_parameters()
    print(f"Model parameters: {params:,} ({params/1e6:.2f}M)")

    criterion = SegmentationLoss(alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_segmenter.pth')
            print(f"   ✅ Best model saved (val_loss={val_loss:.4f})")

        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, output_dir / f'segmenter_epoch_{epoch}.pth')

    print(f"\n✅ Training complete! Best val_loss: {best_loss:.4f}")
    print(f"   Model saved to: {output_dir / 'best_segmenter.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Text Segmenter')
    parser.add_argument('--data-dir', type=str, default='/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/dataset/text_detection',
                        help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='runs/text_ocr/segmenter',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save-freq', type=int, default=10, help='Save frequency')

    args = parser.parse_args()
    main(args)