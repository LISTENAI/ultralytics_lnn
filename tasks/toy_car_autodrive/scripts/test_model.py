#!/usr/bin/env python3
"""
测试模型效果
"""

import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from road_segnet import build_road_segnet
from prepare_dataset import RoadDataset
from torch.utils.data import DataLoader


def test_model(model_path, data_dir, input_size=160, num_tests=20):
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = build_road_segnet(num_classes=3, input_size=input_size)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"模型参数量: {model.count_parameters():,}")
    print(f"最佳IoU: {checkpoint.get('best_iou', 'N/A')}")
    print(f"训练轮次: {checkpoint.get('epoch', 'N/A')}")

    # 加载数据集
    dataset = RoadDataset(data_dir, split='val', input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 测试
    total_iou = 0
    total_acc = 0
    decision_correct = 0

    print(f"\n测试 {num_tests} 个样本...")

    DECISION_NAMES = ['直行', '左转', '右转', '停止']

    for i, (images, masks) in enumerate(dataloader):
        if i >= num_tests:
            break

        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            seg_logit, decision = model(images)

            # 分割结果
            seg_pred = seg_logit.argmax(dim=1)

            # 计算IoU
            pred_np = seg_pred[0].cpu().numpy()
            mask_np = masks[0].cpu().numpy()

            # 道路IoU
            road_pred = (pred_np == 1)
            road_mask = (mask_np == 1)
            intersection = (road_pred & road_mask).sum()
            union = (road_pred | road_mask).sum()
            iou = intersection / (union + 1e-6)
            total_iou += iou

            # 准确率
            acc = (pred_np == mask_np).mean()
            total_acc += acc

        # 生成决策
        road_info = analyze_road(pred_np)
        pred_decision = get_decision(road_info)

        # 打印结果
        print(f"\n样本 {i+1}:")
        print(f"  IoU: {iou:.4f}, 像素准确率: {acc:.4f}")
        print(f"  道路占比: {road_info['road_ratio']:.2%}")
        print(f"  左/中/右: {road_info['left']:.2%} / {road_info['center']:.2%} / {road_info['right']:.2%}")
        print(f"  决策: {DECISION_NAMES[pred_decision]}")

    avg_iou = total_iou / num_tests
    avg_acc = total_acc / num_tests

    print(f"\n{'='*50}")
    print(f"平均IoU: {avg_iou:.4f}")
    print(f"平均像素准确率: {avg_acc:.4f}")
    print(f"{'='*50}")

    return avg_iou, avg_acc


def analyze_road(seg_mask):
    """分析道路"""
    h, w = seg_mask.shape
    road_mask = (seg_mask == 1).astype(np.float32)

    left_road = road_mask[:, :w//3].sum()
    center_road = road_mask[:, w//3:2*w//3].sum()
    right_road = road_mask[:, 2*w//3:].sum()

    total_road = road_mask.sum()
    if total_road == 0:
        return {
            'road_ratio': 0,
            'direction': 'stop',
            'left': 0,
            'center': 0,
            'right': 0
        }

    return {
        'road_ratio': total_road / (h * w),
        'direction': 'center' if center_road > max(left_road, right_road) else
                    'left' if left_road > right_road else 'right',
        'left': left_road / total_road,
        'center': center_road / total_road,
        'right': right_road / total_road
    }


def get_decision(road_info):
    """获取决策"""
    if road_info['road_ratio'] < 0.05:
        return 3  # 停止
    direction_map = {'left': 1, 'center': 0, 'right': 2}
    return direction_map.get(road_info['direction'], 0)


if __name__ == '__main__':
    test_model(
        model_path='runs/toy_car/best_model.pth',
        data_dir='/CodeRepo/Code/dwwang16/dataset/toy_car_road',
        num_tests=30
    )