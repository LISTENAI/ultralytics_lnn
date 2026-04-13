#!/usr/bin/env python3
"""
文本检测、分割、OCR数据集下载器
自动搜索并下载公开的文本检测数据集

支持的数据集:
- ICDAR 2015: 英文文本检测
- ICDAR 2013: 英文文本检测
- Total-Text: 英文文本检测
- SCUT-CTW1500: 中文文本检测

数据集来源:
- 官方ICDAR挑战赛
- GitHub镜像仓库
"""

import os
import sys
import json
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import cv2


# 数据集配置
DATASETS = {
    'icdar2015': {
        'name': 'ICDAR 2015',
        'description': 'Incidental Scene Text Detection',
        'url': 'https://rrc.cvc.uab.es/?ch=4',
        'task': 'Text Detection',
        'languages': ['English'],
        'num_images': 1500,
    },
    'icdar2013': {
        'name': 'ICDAR 2013',
        'description': 'Focused Scene Text Detection',
        'url': 'https://rrc.cvc.uab.es/?ch=2',
        'task': 'Text Detection',
        'languages': ['English'],
        'num_images': 462,
    },
    'totaltext': {
        'name': 'Total-Text',
        'description': 'Text in Natural Scene',
        'url': 'https://github.com/argman/EAST',
        'task': 'Text Detection',
        'languages': ['English'],
        'num_images': 1555,
    },
    'ctw1500': {
        'name': 'SCUT-CTW1500',
        'description': 'Chinese Text in the Wild',
        'url': 'https://github.com/Yuliang-Liu/Curve-Text-Detector',
        'task': 'Text Detection',
        'languages': ['Chinese', 'English'],
        'num_images': 1500,
    }
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """下载文件并显示进度"""
    try:
        print(f"📥 Downloading {desc}: {url[:60]}...")

        # 使用curl下载 (更稳定)
        import subprocess
        result = subprocess.run(
            ['curl', '-L', '-o', str(dest), url],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"   ✅ Downloaded to {dest}")
            return True
        else:
            print(f"   ❌ Download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """解压ZIP文件"""
    try:
        print(f"📦 Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        print(f"   ✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"   ❌ Extract error: {e}")
        return False


def extract_tar(tar_path: Path, extract_to: Path) -> bool:
    """解压TAR文件"""
    try:
        print(f"📦 Extracting {tar_path.name}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        print(f"   ✅ Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"   ❌ Extract error: {e}")
        return False


def download_icdar2015(output_dir: Path) -> bool:
    """下载ICDAR 2015数据集"""
    print("\n" + "=" * 50)
    print("下载 ICDAR 2015 数据集")
    print("=" * 50)

    # 使用GitHub镜像或其他可用源
    base_url = "https://github.com/BuKai2018/icdar2015/archive/refs/heads/master.zip"

    zip_file = output_dir / "icdar2015.zip"

    if download_file(base_url, zip_file, "ICDAR 2015"):
        extract_dir = output_dir / "icdar2015"
        if extract_zip(zip_file, output_dir):
            # 移动文件
            src_dir = output_dir / "icdar2015-master"
            if src_dir.exists():
                if not extract_dir.exists():
                    shutil.move(str(src_dir), str(extract_dir))
                else:
                    shutil.rmtree(src_dir)

            # 清理
            zip_file.unlink()
            return True

    return False


def download_icdar2013(output_dir: Path) -> bool:
    """下载ICDAR 2013数据集"""
    print("\n" + "=" * 50)
    print("下载 ICDAR 2013 数据集")
    print("=" * 50)

    # 使用GitHub镜像
    base_url = "https://github.com/BuKai2018/icdar2013/archive/refs/heads/master.zip"

    zip_file = output_dir / "icdar2013.zip"

    if download_file(base_url, zip_file, "ICDAR 2013"):
        extract_dir = output_dir / "icdar2013"
        if extract_zip(zip_file, output_dir):
            src_dir = output_dir / "icdar2013-master"
            if src_dir.exists():
                if not extract_dir.exists():
                    shutil.move(str(src_dir), str(extract_dir))
                else:
                    shutil.rmtree(src_dir)

            zip_file.unlink()
            return True

    return False


def generate_synthetic_text_dataset(output_dir: Path, num_samples: int = 2000) -> bool:
    """
    生成合成文本数据集
    用于快速测试和训练
    """
    print("\n" + "=" * 50)
    print("生成合成文本数据集")
    print("=" * 50)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 图像和标注目录
    img_dir = output_dir / "images"
    ann_dir = output_dir / "annotations"
    img_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)

    # 中英文字符集
    chinese_chars = [
        '你好世界', '人工智能', '深度学习', '计算机视觉',
        '文本检测', 'OCR识别', '欢迎使用', '机器学习',
        '神经网络', '目标检测', '图像分割', '自动驾驶'
    ]
    english_words = [
        'Hello', 'World', 'AI', 'Deep Learning', 'Computer Vision',
        'Text Detection', 'OCR', 'Welcome', 'Machine Learning',
        'Neural Network', 'Object Detection', 'Image Segmentation'
    ]

    import random
    random.seed(42)

    for i in range(num_samples):
        # 随机生成图像尺寸
        if random.random() < 0.5:
            width, height = 640, 480
        else:
            width, height = 640, 640

        # 创建背景
        bg_color = (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255)
        )
        img = np.ones((height, width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)

        # 随机添加噪点
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 添加文本
        annotations = []

        num_texts = random.randint(1, 5)
        for _ in range(num_texts):
            # 选择文本语言
            if random.random() < 0.3:
                text = random.choice(chinese_chars)
                font_scale = random.uniform(0.8, 1.5)
            else:
                text = random.choice(english_words)
                font_scale = random.uniform(0.6, 1.2)

            # 随机位置
            x = random.randint(20, width - 150)
            y = random.randint(40, height - 40)

            # 随机颜色
            text_color = (
                random.randint(0, 50),
                random.randint(0, 50),
                random.randint(0, 50)
            )

            # 随机字体
            font = random.choice([
                cv2.FONT_HERSHEY_SIMPLEX,
                cv2.FONT_HERSHEY_DUPLEX,
                cv2.FONT_HERSHEY_COMPLEX,
            ])

            # 获取文本尺寸
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, 2
            )

            # 绘制文本
            cv2.putText(img, text, (x, y + text_h), font, font_scale, text_color, 2)

            # 保存标注 (四个角点)
            bbox = [
                [x, y],
                [x + text_w, y],
                [x + text_w, y + text_h],
                [x, y + text_h]
            ]
            annotations.append({
                'text': text,
                'bbox': bbox,
                'language': 'zh' if '\u4e00' <= text[0] <= '\u9fff' else 'en'
            })

        # 保存图像
        img_path = img_dir / f"text_{i:05d}.jpg"
        cv2.imwrite(str(img_path), img)

        # 保存标注
        ann_path = ann_dir / f"text_{i:05d}.json"
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image_id': i,
                'image_path': str(img_path),
                'width': width,
                'height': height,
                'annotations': annotations
            }, f, ensure_ascii=False, indent=2)

        if (i + 1) % 500 == 0:
            print(f"   Generated {i + 1}/{num_samples} images...")

    print(f"   ✅ Generated {num_samples} images")

    # 生成数据集配置
    dataset_config = {
        'name': 'Synthetic Text Dataset',
        'num_images': num_samples,
        'languages': ['Chinese', 'English'],
        'image_dir': str(img_dir),
        'annotation_dir': str(ann_dir),
        'train_split': int(num_samples * 0.8),
        'val_split': int(num_samples * 0.1),
        'test_split': num_samples - int(num_samples * 0.8) - int(num_samples * 0.1)
    }

    config_path = output_dir / "dataset_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, ensure_ascii=False, indent=2)

    print(f"   ✅ Dataset config saved to {config_path}")
    return True


def prepare_dataset(output_dir: Path, split: str = 'train') -> Tuple[List[Path], List[Dict]]:
    """
    准备数据集，返回图像路径和标注列表
    """
    img_dir = output_dir / "images"
    ann_dir = output_dir / "annotations"

    if not img_dir.exists() or not ann_dir.exists():
        return [], []

    # 读取所有标注
    samples = []
    for ann_file in ann_dir.glob("*.json"):
        with open(ann_file, 'r', encoding='utf-8') as f:
            ann = json.load(f)
            img_name = ann_file.stem + '.jpg'
            img_path = img_dir / img_name
            if img_path.exists():
                samples.append({
                    'image_path': img_path,
                    'annotations': ann.get('annotations', [])
                })

    # 划分训练/验证/测试集
    random.shuffle(samples)
    num_samples = len(samples)
    train_size = int(num_samples * 0.8)
    val_size = int(num_samples * 0.1)

    if split == 'train':
        return [s['image_path'] for s in samples[:train_size]], [s['annotations'] for s in samples[:train_size]]
    elif split == 'val':
        return [s['image_path'] for s in samples[train_size:train_size+val_size]], [s['annotations'] for s in samples[train_size:train_size+val_size]]
    else:  # test
        return [s['image_path'] for s in samples[train_size+val_size:]], [s['annotations'] for s in samples[train_size+val_size:]]


def create_dataset_yaml(output_dir: Path):
    """创建YOLO格式数据集配置"""
    # 创建训练/验证/测试目录结构
    for split in ['train', 'val', 'test']:
        split_img_dir = output_dir / split / 'images'
        split_label_dir = output_dir / split / 'labels'
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        # 复制文件
        src_img_dir = output_dir / 'images'
        src_ann_dir = output_dir / 'annotations'

        samples = prepare_dataset(output_dir, split)
        for img_path, anns in zip(*samples):
            # 复制图像
            dest_img = split_img_dir / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            # 转换为YOLO格式标注
            img = Image.open(img_path)
            w, h = img.size

            label_path = split_label_dir / img_path.with_suffix('.txt').name
            with open(label_path, 'w') as f:
                for ann in anns:
                    bbox = ann.get('bbox', [])
                    if len(bbox) == 4:
                        # 计算最小外接矩形
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)

                        # 转换为YOLO格式 (归一化)
                        x_center = ((x_min + x_max) / 2) / w
                        y_center = ((y_min + y_max) / 2) / h
                        width = (x_max - x_min) / w
                        height = (y_max - y_min) / h

                        # 类别0表示文本
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 创建数据集YAML
    yaml_path = output_dir / "text.yaml"
    yaml_content = f"""# Text Detection Dataset
path: {output_dir}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: text
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"   ✅ Dataset YAML saved to {yaml_path}")


def main():
    import random
    random.seed(42)

    # 输出目录
    output_dir = Path("/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/dataset/text_detection")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📚 文本检测数据集下载器")
    print("=" * 60)

    # 首先尝试下载真实数据集，如果失败则生成合成数据集
    print("\n尝试下载真实文本检测数据集...")

    datasets_downloaded = []

    # 尝试下载各数据集
    if download_icdar2015(output_dir):
        datasets_downloaded.append('icdar2015')
        print("✅ ICDAR 2015 下载成功!")

    if download_icdar2013(output_dir):
        datasets_downloaded.append('icdar2013')
        print("✅ ICDAR 2013 下载成功!")

    # 如果真实数据集下载失败，生成合成数据集
    if not datasets_downloaded:
        print("\n⚠️ 真实数据集下载失败，生成合成数据集...")

        # 检查是否有之前的合成数据
        if (output_dir / "images").exists() and (output_dir / "annotations").exists():
            print("   已有合成数据，跳过生成")
        else:
            generate_synthetic_text_dataset(output_dir, num_samples=3000)

        # 准备YOLO格式数据
        create_dataset_yaml(output_dir)
    else:
        # 处理真实数据集
        print(f"\n📊 已下载数据集: {', '.join(datasets_downloaded)}")

    # 统计数据集
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
    print(f"   配置: {output_dir / 'text.yaml'}")


if __name__ == "__main__":
    main()