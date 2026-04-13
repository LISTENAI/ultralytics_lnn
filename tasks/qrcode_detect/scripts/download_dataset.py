#!/usr/bin/env python3
"""
二维码检测数据集下载器/生成器
自动生成合成二维码数据集用于训练

功能:
1. 尝试从网络下载公开二维码数据集
2. 生成合成二维码数据集（主要方式）
"""

import os
import sys
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
import torch

# 数据集输出目录
DEFAULT_DATA_DIR = Path("/CodeRepo/Code/dwwang16/dataset/qrcode_detection")

# 角度映射
ANGLE_CLASSES = {0: 0, 90: 1, 180: 2, 270: 3}


def generate_qr_code(content: str, size: int = 200) -> np.ndarray:
    """
    生成二维码图像

    Args:
        content: 二维码内容
        size: 二维码图像大小

    Returns:
        二维码图像 (灰度图)
    """
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(content)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img = np.array(img)

        # 调整大小
        if size != img.shape[0]:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)

        return img
    except ImportError:
        # 如果没有qrcode库，使用备用方法
        # 创建一个简单的模拟二维码
        img = np.ones((size, size), dtype=np.uint8) * 255

        # 绘制定位图案
        box_size = size // 7

        # 左上角
        cv2.rectangle(img, (box_size, box_size), (box_size * 3, box_size * 3), 0, -1)
        cv2.rectangle(img, (box_size * 2 - box_size // 3, box_size * 2 - box_size // 3),
                     (box_size * 2 + box_size // 3, box_size * 2 + box_size // 3), 255, -1)

        # 右上角
        cv2.rectangle(img, (size - box_size * 3, box_size), (size - box_size, box_size * 3), 0, -1)
        cv2.rectangle(img, (size - box_size * 2 - box_size // 3, box_size * 2 - box_size // 3),
                     (size - box_size * 2 + box_size // 3, box_size * 2 + box_size // 3), 255, -1)

        # 左下角
        cv2.rectangle(img, (box_size, size - box_size * 3), (box_size * 3, size - box_size), 0, -1)
        cv2.rectangle(img, (box_size * 2 - box_size // 3, size - box_size * 2 - box_size // 3),
                     (box_size * 2 + box_size // 3, size - box_size * 2 + box_size // 3), 255, -1)

        return img


def add_noise(img: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """添加噪声"""
    if noise_level > 0:
        noise = np.random.randn(*img.shape) * noise_level * 255
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def add_blur(img: np.ndarray, blur_prob: float = 0.3) -> np.ndarray:
    """添加模糊"""
    if random.random() < blur_prob:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img


def add_occlusion(img: np.ndarray, occ_prob: float = 0.2) -> np.ndarray:
    """添加遮挡"""
    if random.random() < occ_prob:
        h, w = img.shape[:2]
        x = random.randint(0, w - 30)
        y = random.randint(0, h - 30)
        bw = random.randint(10, 30)
        bh = random.randint(10, 30)
        img[y:y+bh, x:x+bw] = random.randint(200, 255)
    return img


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """
    旋转图像

    Args:
        img: 输入图像
        angle: 旋转角度 (0, 90, 180, 270)

    Returns:
        旋转后的图像
    """
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def generate_qr_dataset(output_dir: Path, num_samples: int = 5000,
                        img_size: int = 320) -> bool:
    """
    生成合成二维码数据集

    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        img_size: 图像尺寸

    Returns:
        是否成功
    """
    print("\n" + "=" * 50)
    print("生成合成二维码数据集")
    print("=" * 50)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建目录
    img_dir = output_dir / "images"
    ann_dir = output_dir / "annotations"
    img_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)

    # 二维码内容候选
    qr_contents = [
        f"https://example.com/item/{i}" for i in range(100)
    ] + [
        f"ID:{random.randint(100000, 999999)}" for _ in range(200)
    ] + [
        f"TEXT:{random.choice(['hello', 'world', 'test', 'qrcode', 'data'])}" for _ in range(200)
    ] + [
        f"JSON:{{\"id\":{i},\"type\":\"item\"}}" for i in range(100)
    ]

    random.seed(42)
    np.random.seed(42)

    for i in range(num_samples):
        # 生成二维码内容
        content = random.choice(qr_contents)

        # 生成二维码图像
        qr_size = random.randint(80, 150)  # 二维码在图像中的大小
        qr_img = generate_qr_code(content, size=qr_size)

        # 创建背景图像
        bg_color = random.randint(200, 255)
        img = np.ones((img_size, img_size), dtype=np.uint8) * bg_color

        # 随机位置放置二维码
        max_x = img_size - qr_size - 10
        max_y = img_size - qr_size - 10
        x_offset = random.randint(10, max_x) if max_x > 10 else 10
        y_offset = random.randint(10, max_y) if max_y > 10 else 10

        # 放置二维码
        img[y_offset:y_offset+qr_size, x_offset:x_offset+qr_size] = qr_img

        # 随机旋转 (0, 90, 180, 270)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            # 旋转图像
            img = rotate_image(img, angle)

            # 重新计算二维码位置（用于标注）
            if angle == 90:
                x_offset, y_offset = y_offset, img_size - x_offset - qr_size
            elif angle == 180:
                x_offset = img_size - x_offset - qr_size
                y_offset = img_size - y_offset - qr_size
            elif angle == 270:
                x_offset, y_offset = img_size - y_offset - qr_size, x_offset

        # 添加噪声
        img = add_noise(img, random.uniform(0, 0.15))

        # 添加模糊
        img = add_blur(img, blur_prob=0.2)

        # 添加遮挡
        img = add_occlusion(img, occ_prob=0.15)

        # 保存图像 (转为RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_path = img_dir / f"qr_{i:05d}.jpg"
        cv2.imwrite(str(img_path), img_rgb)

        # 标注
        # 归一化坐标 (YOLO格式)
        x_center = (x_offset + qr_size / 2) / img_size
        y_center = (y_offset + qr_size / 2) / img_size
        width = qr_size / img_size
        height = qr_size / img_size

        # 限制范围
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0.01, 1)
        height = np.clip(height, 0.01, 1)

        ann = {
            'image_id': i,
            'image_path': str(img_path),
            'width': img_size,
            'height': img_size,
            'content': content,
            'angle': angle,
            'bbox': {
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            }
        }

        ann_path = ann_dir / f"qr_{i:05d}.json"
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)

        if (i + 1) % 1000 == 0:
            print(f"   Generated {i + 1}/{num_samples} images...")

    print(f"   ✅ Generated {num_samples} images")

    # 创建数据集配置
    dataset_config = {
        'name': 'Synthetic QR Code Detection Dataset',
        'num_images': num_samples,
        'image_dir': str(img_dir),
        'annotation_dir': str(ann_dir),
        'image_size': img_size,
        'train_split': int(num_samples * 0.8),
        'val_split': int(num_samples * 0.1),
        'test_split': num_samples - int(num_samples * 0.8) - int(num_samples * 0.1)
    }

    config_path = output_dir / "dataset_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, ensure_ascii=False, indent=2)

    print(f"   ✅ Dataset config saved to {config_path}")
    return True


def create_yolo_dataset(output_dir: Path):
    """创建YOLO格式数据集"""
    print("\n" + "=" * 50)
    print("创建YOLO格式数据集")
    print("=" * 50)

    img_dir = output_dir / "images"
    ann_dir = output_dir / "annotations"

    # 创建训练/验证/测试目录
    for split in ['train', 'val', 'test']:
        split_img_dir = output_dir / split / 'images'
        split_label_dir = output_dir / split / 'labels'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

    # 读取所有标注
    samples = []
    for ann_file in sorted(ann_dir.glob("*.json")):
        with open(ann_file, 'r', encoding='utf-8') as f:
            ann = json.load(f)
            img_name = ann_file.stem + '.jpg'
            img_path = img_dir / img_name
            if img_path.exists():
                samples.append({
                    'image_path': img_path,
                    'annotations': ann
                })

    # 随机划分
    random.shuffle(samples)
    num_samples = len(samples)
    train_size = int(num_samples * 0.8)
    val_size = int(num_samples * 0.1)

    splits = {
        'train': samples[:train_size],
        'val': samples[train_size:train_size + val_size],
        'test': samples[train_size + val_size:]
    }

    # 处理每个划分
    for split, split_samples in splits.items():
        for sample in split_samples:
            img_path = sample['image_path']
            ann = sample['annotations']

            # 复制图像
            dest_img = output_dir / split / 'images' / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            # 创建标签文件
            label_path = output_dir / split / 'labels' / img_path.with_suffix('.txt').name

            bbox = ann.get('bbox', {})
            with open(label_path, 'w') as f:
                # YOLO格式: class_id x_center y_center width height
                f.write(f"0 {bbox.get('x_center', 0):.6f} {bbox.get('y_center', 0):.6f} "
                       f"{bbox.get('width', 0):.6f} {bbox.get('height', 0):.6f}\n")

        print(f"   {split}: {len(split_samples)} images")

    # 创建数据集YAML
    yaml_content = f"""# QR Code Detection Dataset
path: {output_dir}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: qrcode
"""
    yaml_path = output_dir / "qrcode.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"   ✅ Dataset YAML saved to {yaml_path}")
    return True


def download_from_online_sources(output_dir: Path) -> bool:
    """
    尝试从网络下载二维码数据集

    Note: 公开的二维码检测数据集较少，主要使用合成数据
    """
    print("\n" + "=" * 50)
    print("尝试下载公开二维码数据集")
    print("=" * 50)

    # 列出可能的公开数据集源
    sources = [
        # ArTeLab QR Dataset - 需要手动下载
        # ("https://...", "artelab_qr"),
    ]

    for url, name in sources:
        print(f"   尝试 {name}...")

    print("   ⚠️ 公开二维码数据集较少，将使用合成数据集")
    return False


def main():
    random.seed(42)

    # 输出目录
    output_dir = DEFAULT_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📚 二维码检测数据集准备")
    print("=" * 60)
    print(f"输出目录: {output_dir}")

    # 尝试在线下载
    download_from_online_sources(output_dir)

    # 检查是否已有数据
    img_dir = output_dir / "images"
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 0:
        print(f"\n⚠️ 数据已存在 ({len(list(img_dir.glob('*.jpg')))} images)")
        response = input("是否重新生成? (y/n): ")
        if response.lower() != 'y':
            print("   使用现有数据")
        else:
            # 删除旧数据
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            generate_qr_dataset(output_dir, num_samples=5000, img_size=320)
    else:
        # 生成合成数据集
        generate_qr_dataset(output_dir, num_samples=5000, img_size=320)

    # 创建YOLO格式数据
    create_yolo_dataset(output_dir)

    # 统计
    print("\n" + "=" * 60)
    print("📈 数据集统计")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        img_dir = output_dir / split / 'images'
        if img_dir.exists():
            num_images = len(list(img_dir.glob('*.jpg')))
            print(f"   {split}: {num_images} images")

    print(f"\n✅ 数据集准备完成!")
    print(f"   数据集路径: {output_dir}")
    print(f"   配置: {output_dir / 'qrcode.yaml'}")


if __name__ == "__main__":
    main()