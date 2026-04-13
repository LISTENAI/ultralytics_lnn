#!/usr/bin/env python3
"""
BDD100K 数据集预处理脚本
下载分割标签并转换为训练可用的格式
"""

import os
import sys
import zipfile
import json
from pathlib import Path
from tqdm import tqdm
import requests
import numpy as np
from PIL import Image
import argparse

# BDD100K 标签下载链接
BDD100K_LABELS_URL = "https://dl.cvlibs.net/datasets/bdd100k/seg/bdd100k_seg_labels_trainval.zip"

# BDD100K 分割类别映射到简化类别
# BDD100K: 0-道路, 1- sidewalk, 2-building, 3-wall, 4-fence, 5-pole, 6-traffic light,
#          7-traffic sign, 8-vegetation, 9-terrain, 10-sky, 11-person, 12-rider,
#          13-car, 14-truck, 15-bus, 16-train, 17-motorcycle, 18-bicycle

# 简化为3类：背景(0)、道路(1)、障碍物(2)
BDD100K_TO_SIMPLE = {
    0: 1,    # road -> 道路
    1: 2,   # sidewalk -> 障碍物（边缘）
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


def download_labels(output_dir):
    """下载BDD100K标签"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "bdd100k_seg_labels.zip"

    if zip_path.exists():
        print("标签文件已存在，跳过下载")
        return zip_path

    print(f"下载 BDD100K 分割标签...")
    print(f"URL: {BDD100K_LABELS_URL}")

    try:
        response = requests.get(BDD100K_LABELS_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f, tqdm(
            desc="下载标签",
            total=total_size,
            unit='B',
            unit_scale=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"标签已下载: {zip_path}")
        return zip_path

    except Exception as e:
        print(f"下载失败: {e}")
        print("\n请手动下载标签:")
        print("  1. 访问: https://bdd-data.berkeley.edu/")
        print("  2. 下载: Segmentation -> Train/Val Labels")
        print(f"  3. 解压到: {output_dir}")
        return None


def convert_labels(zip_path, output_dir):
    """转换标签为简化格式"""
    output_dir = Path(output_dir)
    train_mask_dir = output_dir / "masks" / "train"
    val_mask_dir = output_dir / "masks" / "val"

    train_mask_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)

    print("转换标签...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # 处理验证集
        val_files = [f for f in zf.namelist() if 'val' in f and f.endswith('.png')]

        for name in tqdm(val_files, desc="处理验证集"):
            # 提取文件名
            basename = os.path.basename(name)

            # 读取并转换
            with zf.open(name) as f:
                mask = np.array(Image.open(f))

            # 转换为简化类别
            simple_mask = np.zeros_like(mask)
            for bdd_cls, simple_cls in BDD100K_TO_SIMPLE.items():
                simple_mask[mask == bdd_cls] = simple_cls

            # 保存
            Image.fromarray(simple_mask).save(val_mask_dir / basename)

    print(f"验证集标签已转换: {val_mask_dir}")
    return val_mask_dir


def prepare_bdd100k_data(images_dir, masks_dir, split='train', max_samples=None):
    """准备BDD100K数据集列表"""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")

    if not masks_dir.exists():
        raise FileNotFoundError(f"掩码目录不存在: {masks_dir}")

    # 获取所有图像
    image_files = sorted(list(images_dir.glob('*.jpg')))

    if max_samples:
        image_files = image_files[:max_samples]

    # 过滤有对应掩码的图像
    valid_files = []
    for img_path in image_files:
        mask_path = masks_dir / (img_path.stem + '.png')
        if mask_path.exists():
            valid_files.append((img_path, mask_path))

    print(f"{split}集: 找到 {len(valid_files)} 个有效样本")

    return valid_files


class BDD100KDataset:
    """BDD100K数据集类"""
    def __init__(self, image_files, transform=None, input_size=160):
        self.image_files = image_files
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_files[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 调整大小
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)

        # 转换为tensor
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


def create_dataset_symlink(data_root, bdd100k_dir):
    """创建数据集符号链接"""
    data_root = Path(data_root)
    bdd100k_dir = Path(bdd100k_dir)

    # 链接图像
    images_link = data_root / "images"
    if not images_link.exists():
        # 链接100k/train目录
        src = bdd100k_dir / "100k" / "train"
        if src.exists():
            images_link.symlink_to(src, target_is_directory=True)
            print(f"已创建符号链接: {images_link} -> {src}")

    return images_link


def main():
    parser = argparse.ArgumentParser(description='BDD100K数据集预处理')
    parser.add_argument('--data-dir', type=str,
                       default='/CodeRepo/Code/dwwang16/dataset/toy_car_road',
                       help='数据根目录')
    parser.add_argument('--download-labels', action='store_true',
                       help='下载分割标签')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 下载标签
    if args.download_labels:
        zip_path = download_labels(data_dir)
        if zip_path:
            convert_labels(zip_path, data_dir)

    # 检查数据
    images_dir = data_dir / "100k" / "train"
    masks_dir = data_dir / "masks" / "val"

    if images_dir.exists() and masks_dir.exists():
        files = prepare_bdd100k_data(images_dir, masks_dir, 'val', args.max_samples)
        print(f"\n数据集准备完成: {len(files)} 样本")
    else:
        print("需要下载分割标签并转换")
        print(f"运行: python prepare_bdd100k.py --data-dir {data_dir} --download-labels")


if __name__ == '__main__':
    import torch
    main()