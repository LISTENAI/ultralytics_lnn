#!/usr/bin/env python3
"""
创建更丰富的合成数据集
包含多种道路场景
"""

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def create_curved_road(img_size=160):
    """创建弯曲道路"""
    img = np.random.randint(40, 80, (img_size, img_size, 3), dtype=np.uint8)

    # 添加天空（蓝色渐变）
    for i in range(img_size // 3):
        color = int(100 + i * 2)
        img[i, :, :] = [color, color + 20, color + 50]

    # 添加地面（灰色）
    img[img_size // 3:, :, :] = [80, 80, 80]

    # 创建弯曲道路（正弦曲线）
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    x = np.arange(img_size)
    center = img_size // 2 + np.sin(x * 0.05) * 20

    for i in range(img_size):
        road_width = 20 + int(np.sin(i * 0.03) * 5)
        left = max(0, int(center[i] - road_width))
        right = min(img_size, int(center[i] + road_width))
        mask[i, left:right] = 1  # 道路

    return img, mask


def create_straight_road(img_size=160):
    """创建直路"""
    img = np.random.randint(40, 80, (img_size, img_size, 3), dtype=np.uint8)

    # 天空
    img[:img_size // 4, :, :] = [100, 130, 180]

    # 地面
    img[img_size // 4:, :, :] = [70, 70, 70]

    # 道路（中间）
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    road_width = 25
    mask[:, img_size//2 - road_width:img_size//2 + road_width] = 1

    return img, mask


def create_left_turn_road(img_size=160):
    """创建左转道路"""
    img = np.random.randint(40, 80, (img_size, img_size, 3), dtype=np.uint8)

    # 天空
    img[:img_size // 4, :, :] = [100, 130, 180]
    img[img_size // 4:, :, :] = [70, 70, 70]

    # 左转道路（从中间向左弯曲）
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for i in range(img_size):
        # 道路从中间逐渐向左偏移
        offset = int(i * 0.3)
        left = max(0, img_size // 2 - 15 - offset)
        right = max(0, img_size // 2 + 15 - offset)
        if right > left:
            mask[i, left:right] = 1

    return img, mask


def create_right_turn_road(img_size=160):
    """创建右转道路"""
    img = np.random.randint(40, 80, (img_size, img_size, 3), dtype=np.uint8)

    # 天空
    img[:img_size // 4, :, :] = [100, 130, 180]
    img[img_size // 4:, :, :] = [70, 70, 70]

    # 右转道路
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for i in range(img_size):
        offset = int(i * 0.3)
        left = min(img_size, img_size // 2 - 15 + offset)
        right = min(img_size, img_size // 2 + 15 + offset)
        if right > left:
            mask[i, left:right] = 1

    return img, mask


def create_y_junction(img_size=160):
    """创建Y型路口"""
    img = np.random.randint(40, 80, (img_size, img_size, 3), dtype=np.uint8)

    # 天空
    img[:img_size // 4, :, :] = [100, 130, 180]
    img[img_size // 4:, :, :] = [70, 70, 70]

    # Y型道路
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # 下半部分合并
    for i in range(img_size // 2, img_size):
        mask[i, img_size//2 - 15:img_size//2 + 15] = 1

    # 上半部分分开
    for i in range(img_size // 2):
        ratio = i / (img_size // 2)
        # 两条路逐渐分开
        left = int(img_size // 3 * ratio)
        right = int(img_size // 3 + 15)

        # 左路
        mask[i, left:left + 15] = 1
        # 右路
        mask[i, img_size - right:img_size - right + 15] = 1

    return img, mask


def create_no_road(img_size=160):
    """创建无道路场景（停止）"""
    img = np.random.randint(30, 60, (img_size, img_size, 3), dtype=np.uint8)

    # 天空
    img[:img_size // 3, :, :] = [80, 100, 140]

    # 障碍物区域（灰色砖块）
    img[img_size // 3:, :, :] = [60, 60, 60]

    # 添加一些障碍物纹理
    for i in range(img_size // 3, img_size):
        for j in range(0, img_size, 10):
            img[i, j:j + 5, :] = [50, 50, 50]

    # 没有道路
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    # 全是障碍物
    mask[img_size // 3:, :] = 2  # 障碍物类别

    return img, mask


def create_complex_scene(img_size=160):
    """创建复杂场景（有障碍物）"""
    img, mask = create_straight_road(img_size)

    # 添加障碍物（车辆）
    car_x = img_size // 2 + np.random.randint(-10, 10)
    car_y = img_size // 2 + np.random.randint(-20, 20)
    car_w, car_h = 20, 15

    for i in range(max(0, car_y - car_h//2), min(img_size, car_y + car_h//2)):
        for j in range(max(0, car_x - car_w//2), min(img_size, car_x + car_w//2)):
            img[i, j, :] = [200, 50, 50]  # 红色车辆
            mask[i, j] = 2  # 障碍物

    return img, mask


def generate_dataset(output_dir, num_samples=500):
    """生成完整数据集"""
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images' / 'val'
    masks_dir = output_dir / 'masks' / 'val'

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 场景生成函数和对应数量
    scene_types = [
        (create_straight_road, num_samples // 5),       # 直行
        (create_curved_road, num_samples // 5),          # 弯曲
        (create_left_turn_road, num_samples // 5),       # 左转
        (create_right_turn_road, num_samples // 5),      # 右转
        (create_no_road, num_samples // 10),             # 停止
        (create_y_junction, num_samples // 10),         # Y型路口
        (create_complex_scene, num_samples // 10),     # 有障碍物
    ]

    print(f"生成数据集: {num_samples} 样本")
    idx = 0
    for create_fn, count in scene_types:
        print(f"  生成 {count} 个 {create_fn.__name__}")
        for i in tqdm(range(count)):
            img, mask = create_fn(160)

            # 添加随机噪声
            noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # 保存
            Image.fromarray(img).save(str(images_dir / f'road_{idx:04d}.jpg'))
            Image.fromarray(mask).save(str(masks_dir / f'road_{idx:04d}.png'))
            idx += 1

    print(f"数据集生成完成: {output_dir}")
    print(f"  总样本数: {idx}")
    print(f"  类别分布:")
    print(f"    0 - 背景/障碍物")
    print(f"    1 - 道路")
    print(f"    2 - 障碍物")


if __name__ == '__main__':
    generate_dataset(
        '/CodeRepo/Code/dwwang16/dataset/toy_car_road',
        num_samples=700
    )