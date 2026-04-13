#!/usr/bin/env python3
"""
使用预训练DeepLabV3模型进行BDD100K数据推理测试
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# 类别映射
BDD_TO_SIMPLE = {
    0: 1,   # road -> 道路
    1: 2,   # sidewalk -> 障碍物
    2: 2,   # building
    3: 2,   # wall
    4: 2,   # fence
    5: 2,   # pole
    6: 2,   # traffic light
    7: 2,   # traffic sign
    8: 2,   # vegetation
    9: 0,   # terrain -> 背景
    10: 0,  # sky -> 背景
    11: 2,  # person
    12: 2,  # rider
    13: 2,  # car
    14: 2,  # truck
    15: 2,  # bus
    16: 2,  # train
    17: 2,  # motorcycle
    18: 2,  # bicycle
}

DECISION_NAMES = ['直行', '左转', '右转', '停止']


def load_model():
    """加载预训练DeepLabV3模型"""
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

    print("加载预训练 DeepLabV3...")
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


def process_image(model, image_path, device):
    """处理单张图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)

    # 调整大小以适应模型
    image_resized = image.resize((520, 520), Image.BILINEAR)
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(image_tensor)['out'][0]  # C, H, W

    # 获取预测类别
    pred = output.argmax(dim=0).cpu().numpy()

    # 转换为简单类别
    simple_pred = np.zeros_like(pred)
    for bdd_cls, simple_cls in BDD_TO_SIMPLE.items():
        simple_pred[pred == bdd_cls] = simple_cls

    # 调整回原始尺寸
    simple_pred = Image.fromarray(simple_pred.astype(np.uint8))
    simple_pred = simple_pred.resize(original_size, Image.NEAREST)
    simple_pred = np.array(simple_pred)

    return np.array(image), simple_pred


def analyze_road(seg_mask):
    """分析道路并生成决策"""
    h, w = seg_mask.shape
    road_mask = (seg_mask == 1).astype(np.float32)

    # 三等分
    left = road_mask[:, :w//3].sum()
    center = road_mask[:, w//3:2*w//3].sum()
    right = road_mask[:, 2*w//3:].sum()

    total = road_mask.sum()
    if total == 0:
        return {'road_ratio': 0, 'decision': 3, 'direction': 'stop'}

    left_p = left / total
    center_p = center / total
    right_p = right / total

    # 决策（玩具车视角）
    if total / (h * w) < 0.03:
        decision = 3  # 停止
        direction = 'stop'
    elif center_p > 0.4:
        decision = 0  # 直行
        direction = 'center'
    elif left_p > right_p * 1.3:
        decision = 2  # 右转
        direction = 'right'
    elif right_p > left_p * 1.3:
        decision = 1  # 左转
        direction = 'left'
    else:
        decision = 0  # 直行
        direction = 'balanced'

    return {
        'road_ratio': total / (h * w),
        'decision': decision,
        'direction': direction,
        'left': left_p,
        'center': center_p,
        'right': right_p
    }


def visualize(image, seg_mask, result):
    """可视化结果"""
    # 彩色掩码
    color_map = {
        0: (50, 50, 50),      # 背景 - 深灰
        1: (0, 255, 0),       # 道路 - 绿色
        2: (255, 0, 0)        # 障碍物 - 红色
    }

    colored_mask = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        colored_mask[seg_mask == cls] = color

    # 混合
    alpha = 0.5
    blended = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)

    # 信息
    info = result
    text = f"决策: {DECISION_NAMES[info['decision']]} | 道路: {info['road_ratio']:.1%}"
    cv2.putText(blended, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    sub_text = f"左:{info['left']:.1%} 中:{info['center']:.1%} 右:{info['right']:.1%}"
    cv2.putText(blended, sub_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return blended


def test_bdd100k(model, images_dir, num_samples=20):
    """测试BDD100K数据集"""
    images_dir = Path(images_dir)
    image_files = sorted(list(images_dir.glob('*.jpg')))[:num_samples]

    print(f"\n测试 {len(image_files)} 张图像...")

    device = next(model.parameters()).device

    for i, img_path in enumerate(image_files):
        image, seg_mask = process_image(model, img_path, device)
        result = analyze_road(seg_mask)

        print(f"\n样本 {i+1}: {img_path.name}")
        print(f"  决策: {DECISION_NAMES[result['decision']]}")
        print(f"  道路占比: {result['road_ratio']:.2%}")
        print(f"  左/中/右: {result['left']:.2%} / {result['center']:.2%} / {result['right']:.2%}")

        # 保存可视化结果
        vis = visualize(image, seg_mask, result)
        output_path = str(img_path).replace('100k/train/', 'bdd100k_results/')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str,
                       default='/CodeRepo/Code/dwwang16/dataset/toy_car_road/100k/train',
                       help='图像目录')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='测试样本数')
    args = parser.parse_args()

    # 加载模型
    model = load_model()

    # 测试
    test_bdd100k(model, args.images_dir, args.num_samples)


if __name__ == '__main__':
    main()