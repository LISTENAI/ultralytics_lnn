#!/usr/bin/env python3
"""
直接使用训练好的模型测试BDD100K数据集
无需额外标签，使用模型自身的分割能力
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from road_segnet import build_road_segnet


DECISION_NAMES = ['直行', '左转', '右转', '停止']


class BDD100KTester:
    """BDD100K 数据集测试器"""
    def __init__(self, model_path, device='cuda', input_size=160):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = build_road_segnet(num_classes=2, input_size=input_size)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        params = self.model.count_parameters()
        print(f"参数量: {params:,} ({params/1e6:.2f}M)")

    def preprocess(self, image_path):
        """预处理图像"""
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.original_size = (image.shape[1], image.shape[0])

        # 调整大小
        image_resized = cv2.resize(image, (self.input_size, self.input_size))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor, image

    def infer(self, image_tensor):
        """推理"""
        with torch.no_grad():
            seg_logit, decision = self.model(image_tensor)
            seg_pred = seg_logit.argmax(dim=1)[0]  # H, W
            decision_prob = F.softmax(decision, dim=1)[0]

        return seg_pred.cpu().numpy(), decision_prob.cpu().numpy()

    def analyze_road(self, seg_mask):
        """分析道路"""
        h, w = seg_mask.shape
        road_mask = (seg_mask == 1).astype(np.float32)

        left = road_mask[:, :w//3].sum()
        center = road_mask[:, w//3:2*w//3].sum()
        right = road_mask[:, 2*w//3:].sum()

        total = road_mask.sum()
        if total == 0:
            return {
                'road_ratio': 0,
                'decision': 3,
                'direction': 'stop',
                'left': 0, 'center': 0, 'right': 0
            }

        left_p = left / total
        center_p = center / total
        right_p = right / total

        # 决策（玩具车视角）
        if total / (h * w) < 0.03:
            decision = 3
            direction = 'stop'
        elif left_p > right_p * 1.3:  # 图像左侧道路多 -> 右转
            decision = 2
            direction = 'right'
        elif right_p > left_p * 1.3:  # 图像右侧道路多 -> 左转
            decision = 1
            direction = 'left'
        elif center_p > 0.3:
            decision = 0
            direction = 'center'
        else:
            decision = 0
            direction = 'balanced'

        return {
            'road_ratio': total / (h * w),
            'decision': decision,
            'direction': direction,
            'left': left_p,
            'center': center_p,
            'right': right_p
        }

    def visualize(self, original_image, seg_mask, result):
        """可视化"""
        seg_mask_resized = cv2.resize(
            seg_mask.astype(np.uint8),
            self.original_size,
            interpolation=cv2.INTER_NEAREST
        )

        color_map = {
            0: (50, 50, 50),      # 背景 - 深灰
            1: (0, 255, 0),       # 道路 - 绿色
            2: (255, 0, 0)        # 障碍物 - 红色
        }

        colored_mask = np.zeros((*seg_mask_resized.shape, 3), dtype=np.uint8)
        for cls, color in color_map.items():
            colored_mask[seg_mask_resized == cls] = color

        alpha = 0.5
        blended = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)

        # 信息
        info = result
        text = f"决策: {DECISION_NAMES[info['decision']]} | 道路: {info['road_ratio']:.1%}"
        cv2.putText(blended, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        sub_text = f"左:{info['left']:.1%} 中:{info['center']:.1%} 右:{info['right']:.1%}"
        cv2.putText(blended, sub_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return blended


def test_bdd100k(model_path, images_dir, num_samples=50, output_dir='bdd100k_results'):
    """测试BDD100K数据集"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图像
    image_files = sorted(list(images_dir.glob('*.jpg')))[:num_samples]

    print(f"找到 {len(image_files)} 张图像")

    # 创建测试器
    tester = BDD100KTester(model_path)

    # 统计
    decisions = {0: 0, 1: 0, 2: 0, 3: 0}
    road_ratios = []

    print(f"\n测试 {len(image_files)} 张图��...")

    for i, img_path in enumerate(tqdm(image_files)):
        image_tensor, original_image = tester.preprocess(str(img_path))

        if image_tensor is None:
            continue

        seg_pred, decision_prob = tester.infer(image_tensor)
        result = tester.analyze_road(seg_pred)

        # 统计
        decisions[result['decision']] += 1
        road_ratios.append(result['road_ratio'])

        # 保存结果
        vis = tester.visualize(original_image, seg_pred, result)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # 打印结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    print(f"总样本数: {len(image_files)}")
    print(f"\n决策分布:")
    print(f"  直行: {decisions[0]} ({decisions[0]/len(image_files)*100:.1f}%)")
    print(f"  左转: {decisions[1]} ({decisions[1]/len(image_files)*100:.1f}%)")
    print(f"  右转: {decisions[2]} ({decisions[2]/len(image_files)*100:.1f}%)")
    print(f"  停止: {decisions[3]} ({decisions[3]/len(image_files)*100:.1f}%)")
    print(f"\n道路占比统计:")
    print(f"  平均: {np.mean(road_ratios):.2%}")
    print(f"  最小: {np.min(road_ratios):.2%}")
    print(f"  最大: {np.max(road_ratios):.2%}")
    print(f"\n结果已保存: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/toy_car/best_model.pth',
                       help='模型路径')
    parser.add_argument('--images-dir', type=str,
                       default='/CodeRepo/Code/dwwang16/dataset/toy_car_road/100k/train',
                       help='图像目录')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='测试样本数')
    parser.add_argument('--output', type=str, default='bdd100k_results',
                       help='输出目录')
    args = parser.parse_args()

    test_bdd100k(args.model, args.images_dir, args.num_samples, args.output)


if __name__ == '__main__':
    main()