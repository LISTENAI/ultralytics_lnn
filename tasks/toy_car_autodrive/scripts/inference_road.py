#!/usr/bin/env python3
"""
优化版推理脚本
改进决策逻辑
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from road_segnet import build_road_segnet


DECISION_NAMES = ['直行', '左转', '右转', '停止']


class RoadSegInference:
    """道路分割推理类"""
    def __init__(self, model_path, device='cuda', input_size=160):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

        # 加载模型
        self.model = build_road_segnet(num_classes=3, input_size=input_size)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"模型加载成功")
        print(f"参数量: {self.model.count_parameters():,} (目标<4M)")

    def preprocess(self, image):
        """预处理图像"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        self.original_size = (image.shape[1], image.shape[0])
        image_resized = cv2.resize(image, (self.input_size, self.input_size))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor, image

    def infer(self, image_tensor):
        """推理"""
        with torch.no_grad():
            seg_logit, decision = self.model(image_tensor)
            seg_pred = seg_logit.argmax(dim=1)[0]
            decision_prob = F.softmax(decision, dim=1)[0]
            decision_pred = decision[0].argmax().item()

        return seg_pred, decision_pred, decision_prob

    def analyze_road(self, seg_mask):
        """分析道路状况 - 改进版"""
        h, w = seg_mask.shape
        road_mask = (seg_mask == 1).cpu().numpy().astype(np.float32)

        # 三等分
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
                'right': 0,
                'decision': 3
            }

        left_p = left_road / total_road
        center_p = center_road / total_road
        right_p = right_road / total_road

        # 改进的决策逻辑 - 玩具车视角（底部往上看）
        # 图像坐标系与车视角相反：
        # 图像左侧道路(left_p) -> 车的右侧 -> 需要右转
        # 图像右侧道路(right_p) -> 车的左侧 -> 需要左转

        if total_road / (h * w) < 0.03:  # 道路太少
            decision = 3  # 停止
            direction = 'stop'
        elif center_p > 0.4:  # 中间道路明显
            decision = 0  # 直行
            direction = 'center'
        elif left_p > right_p * 1.3:  # 图像左侧道路多 -> 右转
            decision = 2
            direction = 'right'
        elif right_p > left_p * 1.3:  # 图像右侧道路多 -> 左转
            decision = 1
            direction = 'left'
        else:  # 均衡
            decision = 0  # 直行
            direction = 'balanced'

        return {
            'road_ratio': total_road / (h * w),
            'direction': direction,
            'left': left_p,
            'center': center_p,
            'right': right_p,
            'decision': decision
        }

    def get_decision(self, seg_mask):
        """获取决策"""
        road_info = self.analyze_road(seg_mask)
        return road_info['decision']

    def process_image(self, image_path):
        """处理单张图像"""
        image_tensor, original_image = self.preprocess(image_path)
        seg_pred, decision_pred, decision_prob = self.infer(image_tensor)

        # 获取决策（使用改进的逻辑）
        decision = self.get_decision(seg_pred)

        result = {
            'decision': DECISION_NAMES[decision],
            'decision_id': decision,
            'confidence': decision_prob[decision].item(),
            'road_info': self.analyze_road(seg_pred)
        }

        return result, seg_pred.cpu().numpy(), original_image

    def visualize(self, original_image, seg_mask, result):
        """可视化结果"""
        seg_mask_resized = cv2.resize(
            seg_mask.astype(np.uint8),
            self.original_size,
            interpolation=cv2.INTER_NEAREST
        )

        # 彩色掩码
        color_map = {
            0: (50, 50, 50),      # 背景
            1: (0, 255, 0),       # 道路 - 绿色
            2: (255, 0, 0)        # 障碍物 - 红色
        }

        colored_mask = np.zeros((*seg_mask_resized.shape, 3), dtype=np.uint8)
        for cls, color in color_map.items():
            colored_mask[seg_mask_resized == cls] = color

        # 混合显示
        alpha = 0.5
        blended = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)

        # 决策信息
        info = result['road_info']
        decision_text = f"决策: {result['decision']}"
        cv2.putText(blended, decision_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        road_text = f"道路: {info['road_ratio']:.1%} | 左:{info['left']:.1%} 中:{info['center']:.1%} 右:{info['right']:.1%}"
        cv2.putText(blended, road_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return blended


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/toy_car_autodrive/best_model.pth')
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    inference = RoadSegInference(args.model)
    result, seg_mask, orig_img = inference.process_image(args.image)

    print("\n" + "=" * 50)
    print("推理结果:")
    print(f"  决策: {result['decision']}")
    print(f"  道路占比: {result['road_info']['road_ratio']:.2%}")
    print(f"  左/中/右: {result['road_info']['left']:.2%} / {result['road_info']['center']:.2%} / {result['road_info']['right']:.2%}")
    print("=" * 50)

    # 保存结果
    vis = inference.visualize(orig_img, seg_mask, result)
    output_path = args.image.replace('.', '_result.')
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"\n可视化结果已保存: {output_path}")


if __name__ == '__main__':
    main()