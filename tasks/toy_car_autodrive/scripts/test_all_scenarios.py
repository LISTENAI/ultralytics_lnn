#!/usr/bin/env python3
"""
批量测试脚本 - 测试所有场景类型
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from road_segnet import build_road_segnet
from prepare_dataset import RoadDataset
from torch.utils.data import DataLoader


def test_all_scenarios(model_path, data_dir):
    """测试所有场景类型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = build_road_segnet(num_classes=3, input_size=160)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载数据
    dataset = RoadDataset(data_dir, split='val', input_size=160)

    # 按场景类型测试
    scenarios = {
        '直行道路 (0-139)': list(range(0, 140)),
        '弯曲道路 (140-279)': list(range(140, 280)),
        '左转道路 (280-419)': list(range(280, 420)),
        '右转道路 (420-559)': list(range(420, 560)),
        '停止场景 (560-629)': list(range(560, 630)),
        'Y型路口 (630-699)': list(range(630, 700)),
    }

    print("="*60)
    print("场景测试结果")
    print("="*60)

    DECISION_NAMES = ['直行', '左转', '右转', '停止']

    for scenario_name, indices in scenarios.items():
        total_iou = 0
        decisions = {0: 0, 1: 0, 2: 0, 3: 0}

        for idx in indices[:20]:  # 每类测试20个样本
            img, mask = dataset[idx]
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                seg_logit, _ = model(img)
                seg_pred = seg_logit.argmax(dim=1)[0]

            # 计算IoU
            pred_np = seg_pred.cpu().numpy()
            mask_np = mask.numpy()
            road_pred = (pred_np == 1)
            road_mask = (mask_np == 1)
            intersection = (road_pred & road_mask).sum()
            union = (road_pred | road_mask).sum()
            iou = intersection / (union + 1e-6)
            total_iou += iou

            # 决策 - 从玩具车视角（底部往上看，图像坐标系与车视角相反）
            # 图像左侧道路多(left) -> 车需要右转
            # 图像右侧道路多(right) -> 车需要左转
            road_mask_float = (pred_np == 1).astype(np.float32)
            h, w = pred_np.shape
            left = road_mask_float[:, :w//3].sum()
            center = road_mask_float[:, w//3:2*w//3].sum()
            right = road_mask_float[:, 2*w//3:].sum()

            if left > right * 1.3:  # 图像左侧道路多 -> 右转
                decision = 2
            elif right > left * 1.3:  # 图像右侧道路多 -> 左转
                decision = 1
            elif center > max(left, right) * 0.5:
                decision = 0
            else:
                decision = 0
            decisions[decision] += 1

        avg_iou = total_iou / min(20, len(indices))
        majority_decision = max(decisions, key=decisions.get)

        print(f"\n{scenario_name}:")
        print(f"  平均IoU: {avg_iou:.4f}")
        print(f"  决策分布: 直行={decisions[0]} 左转={decisions[1]} 右转={decisions[2]} 停止={decisions[3]}")
        print(f"  主要决策: {DECISION_NAMES[majority_decision]}")


if __name__ == '__main__':
    test_all_scenarios(
        'runs/toy_car/best_model.pth',
        '/CodeRepo/Code/dwwang16/dataset/toy_car_road'
    )