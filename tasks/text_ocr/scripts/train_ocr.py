#!/usr/bin/env python3
"""
OCR识别模型训练脚本
训练轻量级CRNN网络 (支持中英文识别)
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
from text_models import CRNN


# 中英文字符集
CHAR_SET_EN = '0123456789abcdefghijklmnopqrstuvwxyz'
CHAR_SET_CN = '的一是不了人我在有他这为之大来以个中上们'
CHAR_SET_PUNCT = ' .,!?;:+-*/%#$@()[]{}<>=\'"\\|~^'

FULL_CHAR_SET = CHAR_SET_EN + CHAR_SET_CN + CHAR_SET_PUNCT


class OCRDataset(Dataset):
    """OCR数据集"""
    def __init__(self, data_dir: Path, split: str = 'train',
                 img_height: int = 32, img_width: int = 128,
                 max_text_len: int = 32):
        self.data_dir = data_dir
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_len = max_text_len

        # 字符表
        self.char_to_idx = {c: i for i, c in enumerate(FULL_CHAR_SET)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.num_classes = len(FULL_CHAR_SET) + 1  # +1 for blank

        # 加载数据
        img_dir = data_dir / split / 'images'
        ann_dir = data_dir / split / 'annotations'

        self.samples = []
        for ann_file in ann_dir.glob('*.json'):
            with open(ann_file, 'r', encoding='utf-8') as f:
                ann = json.load(f)
                img_path = img_dir / ann_file.stem.replace('text_', 'text_') + '.jpg'

                if img_path.exists():
                    for text_ann in ann.get('annotations', []):
                        text = text_ann.get('text', '')
                        if 1 <= len(text) <= max_text_len:
                            self.samples.append({
                                'image_path': img_path,
                                'text': text
                            })

        print(f"Loaded {len(self.samples)} {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 读取图像
        img = cv2.imread(str(sample['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 裁剪文本区域 (简化为中心裁剪)
        h, w = img.shape[:2]
        crop_w = min(w, h * 3)
        x1 = (w - crop_w) // 2
        img = img[:, x1:x1+crop_w]

        # 调整大小
        img = cv2.resize(img, (self.img_width, self.img_height))

        # 灰度化
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 归一化
        img = img.astype(np.float32) / 255.0

        # 添加通道维度
        img = torch.from_numpy(img).unsqueeze(0)

        # 文本编码
        text = sample['text']
        text_encoded = [self.char_to_idx.get(c, 0) for c in text]
        text_length = len(text_encoded)

        # Padding
        text_padded = text_encoded + [self.num_classes - 1] * (self.max_text_len - len(text_encoded))

        return img, torch.tensor(text_padded), torch.tensor(text_length)


class CRNNLoss(nn.Module):
    """CRNN CTC Loss"""
    def __init__(self):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def forward(self, log_probs, targets, target_lengths):
        # log_probs: (T, B, C)
        log_probs = log_probs.log_softmax(2)

        input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for imgs, texts, text_lengths in pbar:
        imgs = imgs.to(device)

        # 转换格式: (T, B, C)
        outputs = model(imgs)
        T, B, C = outputs.size()
        outputs = outputs.view(T, B, C)

        # 计算损失
        loss = criterion(outputs, texts.to(device), text_lengths.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, idx_to_char):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, texts, text_lengths in tqdm(dataloader, desc='Validating'):
            imgs = imgs.to(device)

            outputs = model(imgs)
            T, B, C = outputs.size()
            outputs = outputs.view(T, B, C)

            loss = criterion(outputs, texts.to(device), text_lengths.to(device))
            total_loss += loss.item()

            # 解码预测结果
            _, max_idx = outputs.max(2)
            max_idx = max_idx.permute(1, 0).cpu().numpy()

            # CTC解码 (简化版)
            for b in range(B):
                pred = []
                prev = -1
                for t in range(len(max_idx[b])):
                    idx = max_idx[b][t]
                    if idx != prev and idx < len(idx_to_char):
                        pred.append(idx)
                    prev = idx

                # 比较
                target = texts[b].cpu().numpy()[:text_lengths[b]]
                if len(pred) > 0 and len(pred) == len(target):
                    if all(p == t for p, t in zip(pred, target)):
                        correct += 1
                total += 1

    accuracy = correct / max(total, 1)
    return total_loss / len(dataloader), accuracy


def decode_prediction(outputs, idx_to_char):
    """CTC解码"""
    _, max_idx = outputs.max(2)
    max_idx = max_idx.permute(1, 0).cpu().numpy()

    results = []
    for pred in max_idx:
        text = []
        prev = -1
        for idx in pred:
            if idx != prev and idx < len(idx_to_char):
                text.append(idx_to_char[idx])
            prev = idx
        results.append(''.join(text))
    return results


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_dataset = OCRDataset(data_dir, 'train', args.img_height, args.img_width)
    val_dataset = OCRDataset(data_dir, 'val', args.img_height, args.img_width)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CRNN(num_classes=train_dataset.num_classes).to(device)
    params = model.count_parameters()
    print(f"Model parameters: {params:,} ({params/1e6:.2f}M)")

    criterion = CRNNLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存字符表
    char_map = {
        'char_to_idx': train_dataset.char_to_idx,
        'idx_to_char': train_dataset.idx_to_char,
        'num_classes': train_dataset.num_classes
    }
    with open(output_dir / 'char_map.json', 'w', encoding='utf-8') as f:
        json.dump(char_map, f, ensure_ascii=False)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, accuracy = validate(model, val_loader, criterion, device, train_dataset.idx_to_char)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={accuracy:.2%}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'accuracy': accuracy,
            }, output_dir / 'best_ocr.pth')
            print(f"   ✅ Best model saved (val_loss={val_loss:.4f})")

        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, output_dir / f'ocr_epoch_{epoch}.pth')

    print(f"\n✅ Training complete! Best val_loss: {best_loss:.4f}")
    print(f"   Model saved to: {output_dir / 'best_ocr.pth'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OCR')
    parser.add_argument('--data-dir', type=str, default='/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/dataset/text_detection',
                        help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='runs/text_ocr/ocr',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img-height', type=int, default=32, help='Image height')
    parser.add_argument('--img-width', type=int, default=128, help='Image width')
    parser.add_argument('--max-text-len', type=int, default=32, help='Max text length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save-freq', type=int, default=10, help='Save frequency')

    args = parser.parse_args()
    main(args)