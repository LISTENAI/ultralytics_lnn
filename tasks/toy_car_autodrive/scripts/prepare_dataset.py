#!/usr/bin/env python3
"""
自动驾驶玩具车数据集准备脚本

自动下载并准备道路分割数据集
使用 BDD100K 数据集的分割子集
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm

# 数据集URL (BDD100K 分割验证集)
BDD100K_URL = "https://dl.cvlibs.net/datasets/bdd100k/seg/bdd100k_seg_val.zip"
BDD100K_INFO_URL = "https://dl.cvlibs.net/datasets/bdd100k/seg/bdd100k_seg_labels_trainval.zip"

# 类别定义
# BDD100K 道路分割类别
CLASSES = [
    'road', 'sky', 'building', 'wall', 'fence',
    'vegetation', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle', 'person', 'rider', 'traffic light',
    'traffic sign', 'terrain'
]

# 简化的类别（玩具车关注道路和可行驶区域）
SIMPLE_CLASSES = {
    0: 'background',  # 其他
    1: 'road',         # 道路
    2: 'obstacle',     # 障碍物（车辆、行人、建筑物等）
}


class RoadDataset(Dataset):
    """道路分割数据集"""
    def __init__(self, root_dir, split='val', transform=None, input_size=160):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.input_size = input_size

        # 查找所有图像
        self.images_dir = self.root_dir / 'images' / split
        self.masks_dir = self.root_dir / 'masks' / split

        if not self.images_dir.exists():
            raise FileNotFoundError(f"数据集不存在: {self.images_dir}")

        # 获取所有图像文件
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        print(f"找到 {len(self.image_files)} 张图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # stem 是文件名不含扩展名，直接加 .png
        mask_path = self.masks_dir / (img_path.stem + '.png')

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 调整大小
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)

        # 转换为tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # 简化mask：道路(0)->1, 障碍物(其他)->2, 背景->0
        mask_np = np.array(mask)
        simple_mask = np.zeros_like(mask_np)
        simple_mask[mask_np == 0] = 1  # 道路
        simple_mask[mask_np > 0] = 2   # 障碍物

        mask = torch.from_numpy(simple_mask).long()

        return image, mask


def download_file(url, dest_path, desc='下载'):
    """下载文件并显示进度"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def prepare_bdd100k_dataset(data_dir):
    """准备BDD100K数据集"""
    data_dir = Path(data_dir)

    # 检查是否已下载
    if (data_dir / 'images').exists() and (data_dir / 'masks').exists():
        print("数据集已存在，跳过下载")
        return True

    print("=" * 60)
    print("准备下载 BDD100K 道路分割数据集...")
    print("=" * 60)

    # 创建临时目录
    temp_dir = data_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 下载验证集
        print("\n[1/2] 下载图像数据...")
        zip_path = temp_dir / 'bdd100k_seg_val.zip'
        if not zip_path.exists():
            download_file(BDD100K_URL, zip_path, '下载验证集')
        else:
            print("验证集已存在")

        # 解压
        print("\n[2/2] 解压数据...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)

        # 移动文件到目标位置
        val_images_dir = temp_dir / 'bdd100k' / 'seg' / 'images' / 'val'
        val_masks_dir = temp_dir / 'bdd100k' / 'seg' / 'masks' / 'val'

        # 创建目标目录
        images_dir = data_dir / 'images'
        masks_dir = data_dir / 'masks'
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / 'val').mkdir(exist_ok=True)
        (masks_dir / 'val').mkdir(exist_ok=True)

        # 移动文件
        if val_images_dir.exists():
            print("复制图像文件...")
            for img in tqdm(val_images_dir.glob('*.jpg')):
                import shutil
                shutil.copy(img, images_dir / 'val' / img.name)

        if val_masks_dir.exists():
            print("复制掩码文件...")
            for mask in tqdm(val_masks_dir.glob('*.png')):
                import shutil
                shutil.copy(mask, masks_dir / 'val' / mask.name)

        print("\n数据集准备完成!")
        return True

    except Exception as e:
        print(f"下载失败: {e}")
        print("\n请手动下载数据集:")
        print(f"  1. 访问: https://bdd-data.berkeley.edu/")
        print(f"  2. 下载: BDD100K Segmentation")
        print(f"  3. 解压到: {data_dir}")
        return False
    finally:
        # 清理临时文件
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def create_synthetic_dataset(data_dir, num_samples=1000, input_size=160):
    """
    创建合成数据集（用于测试）
    当无法下载真实数据时使用
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images' / 'val'
    masks_dir = data_dir / 'masks' / 'val'

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"创建合成数据集: {num_samples} 样本")

    np.random.seed(42)
    for i in tqdm(range(num_samples)):
        # 创建随机道路图像
        img = np.random.randint(50, 200, (input_size, input_size, 3), dtype=np.uint8)

        # 添加一些道路特征（灰色条纹）
        for j in range(0, input_size, 20):
            img[j:j+5, :, :] = [100, 100, 100]

        # 创建掩码
        mask = np.zeros((input_size, input_size), dtype=np.uint8)
        # 道路区域（中间部分）
        mask[:, input_size//4:3*input_size//4] = 0  # 道路
        mask[:, :input_size//4] = 1  # 障碍物
        mask[:, 3*input_size//4:] = 1  # 障碍物

        # 保存
        Image.fromarray(img).save(str(images_dir / f'road_{i:04d}.jpg'))
        Image.fromarray(mask).save(str(masks_dir / f'road_{i:04d}.png'))

    print(f"合成数据集创建完成: {data_dir}")


def main():
    parser = argparse.ArgumentParser(description='准备自动驾驶玩具车数据集')
    parser.add_argument('--data-dir', type=str,
                       default='dataset/toy_car_road',
                       help='数据集目录')
    parser.add_argument('--synthetic', action='store_true',
                       help='创建合成数据集（用于测试）')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='合成数据集样本数')
    parser.add_argument('--input-size', type=int, default=160,
                        help='输入图像尺寸')

    args = parser.parse_args()

    if args.synthetic:
        # 创建合成数据集
        create_synthetic_dataset(args.data_dir, args.num_samples, args.input_size)
    else:
        # 尝试下载真实数据集，失败则创建合成数据集
        success = prepare_bdd100k_dataset(args.data_dir)
        if not success:
            print("\n自动下载失败，创建合成数据集用于演示...")
            create_synthetic_dataset(args.data_dir, args.num_samples, args.input_size)


if __name__ == '__main__':
    main()