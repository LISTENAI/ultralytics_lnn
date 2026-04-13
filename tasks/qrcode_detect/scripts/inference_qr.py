#!/usr/bin/env python3
"""
二维码检测与识别推理脚本
完整流水线: 检测 -> 校正 -> 解码
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import json

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from qrcode_models import QRDetector


class QRCodeDetector:
    """二维码检测器推理类"""
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = QRDetector(in_channels=3, num_angle_classes=4)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 角度映射
        self.angle_map = {0: 0, 1: 90, 2: 180, 3: 270}

        # 二维码解码器
        self.qr_decoder = cv2.QRCodeDetector()

        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")

    def preprocess(self, img: np.ndarray, img_size: int = 320) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        img_resized = cv2.resize(img, (img_size, img_size))

        # 归一化
        img_normalized = img_resized.astype(np.float32) / 255.0

        # 转换为tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度

        return img_tensor.to(self.device)

    def detect(self, img: np.ndarray, threshold: float = 0.5) -> dict:
        """
        检测二维码

        Args:
            img: 输入图像 (RGB)
            threshold: 检测阈值

        Returns:
            检测结果字典
        """
        with torch.no_grad():
            img_tensor = self.preprocess(img)
            outputs = self.model(img_tensor)

            # 提取结果
            prob_map = outputs['prob_map'][0, 0].cpu().numpy()
            angle_logits = outputs['angle_logits'][0].cpu().numpy()
            bbox_offset = outputs['bbox_offset'][0].cpu().numpy()

            # 预测角度
            angle_idx = np.argmax(angle_logits)
            angle = self.angle_map[angle_idx]
            angle_conf = float(torch.softmax(torch.tensor(angle_logits), dim=0).max().item())

            # 从概率图提取边界框
            bboxes = self._extract_bboxes(prob_map, threshold)

            return {
                'prob_map': prob_map,
                'bboxes': bboxes,
                'angle': angle,
                'angle_conf': angle_conf,
                'angle_logits': angle_logits
            }

    def _extract_bboxes(self, prob_map: np.ndarray, threshold: float = 0.5) -> list:
        """从概率图提取边界框"""
        # 二值化
        binary = (prob_map > threshold).astype(np.uint8)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # 过滤小区域
                rect = cv2.minAreaRect(cnt)
                bbox = cv2.boxPoints(rect)
                bboxes.append({
                    'points': bbox,
                    'area': area,
                    'rect': rect
                })

        return bboxes

    def decode_qr(self, img: np.ndarray, bbox: dict) -> dict:
        """
        解码二维码

        Args:
            img: 输入图像 (RGB)
            bbox: 检测到的边界框

        Returns:
            解码结果
        """
        # 提取二维码区域
        points = bbox['points']
        points = np.array(points, dtype=np.int32)

        # 获取ROI
        x, y, w, h = cv2.boundingRect(points)
        x = max(0, x - 10)
        y = max(0, y - 10)
        w = min(img.shape[1] - x, w + 20)
        h = min(img.shape[0] - y, h + 20)

        roi = img[y:y+h, x:x+w]

        # 转为灰度
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi

        # 尝试直接解码
        try:
            ret, decoded_info, points, straight_qrcode = self.qr_decoder.detectAndDecodeMulti(gray)
            if ret and decoded_info:
                return {
                    'success': True,
                    'data': decoded_info[0] if decoded_info else "",
                    'points': points
                }
        except:
            pass

        # 尝试多种预处理
        for method in ['normal', 'enhance', 'threshold']:
            try:
                if method == 'normal':
                    test_img = gray
                elif method == 'enhance':
                    test_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
                else:
                    _, test_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                ret, decoded_info, points, _ = self.qr_decoder.detectAndDecodeMulti(test_img)
                if ret and decoded_info:
                    return {
                        'success': True,
                        'data': decoded_info[0] if decoded_info else "",
                        'points': points
                    }
            except:
                continue

        return {
            'success': False,
            'data': '',
            'points': None
        }

    def process(self, img: np.ndarray, threshold: float = 0.5) -> dict:
        """
        完整处理流程: 检测 -> 解码

        Args:
            img: 输入图像 (RGB)
            threshold: 检测阈值

        Returns:
            处理结果
        """
        # 检测
        detection = self.detect(img, threshold)

        results = {
            'has_qrcode': len(detection['bboxes']) > 0,
            'angle': detection['angle'],
            'angle_conf': detection['angle_conf'],
            'bboxes': [],
            'decoded': []
        }

        # 对每个检测框尝试解码
        for bbox in detection['bboxes']:
            # 转换为原始图像坐标
            scale_x = img.shape[1] / 320
            scale_y = img.shape[0] / 320
            points = bbox['points'].copy()
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y

            # 解码
            decode_result = self.decode_qr(img, bbox)

            results['bboxes'].append({
                'points': points.tolist(),
                'area': bbox['area']
            })
            results['decoded'].append(decode_result)

        return results


def visualize_result(img: np.ndarray, result: dict) -> np.ndarray:
    """可视化结果"""
    vis_img = img.copy()

    # 绘制边界框
    for bbox in result.get('bboxes', []):
        points = np.array(bbox['points'], dtype=np.int32)
        cv2.polylines(vis_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 绘制角度信息
    if result.get('has_qrcode'):
        angle = result.get('angle', 0)
        conf = result.get('angle_conf', 0)
        text = f"Angle: {angle}° ({conf:.2f})"
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)

    # 绘制解码结果
    for i, decoded in enumerate(result.get('decoded', [])):
        y_pos = 70 + i * 30
        if decoded['success']:
            text = f"QR{i}: {decoded['data'][:30]}"
            color = (0, 255, 0)
        else:
            text = f"QR{i}: Decode failed"
            color = (0, 0, 255)
        cv2.putText(vis_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)

    return vis_img


def process_image(detector: QRCodeDetector, image_path: str, output_path: str = None):
    """处理单张图像"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 处理
    result = detector.process(img)

    # 可视化
    vis_img = visualize_result(img, result)

    # 保存结果
    if output_path:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_img)
        print(f"Result saved to {output_path}")

    # 打印结果
    print("\n" + "=" * 50)
    print("检测结果")
    print("=" * 50)
    print(f"是否包含二维码: {result['has_qrcode']}")
    print(f"检测数量: {len(result['bboxes'])}")
    print(f"预测角度: {result['angle']}° (置信度: {result['angle_conf']:.2f})")

    for i, decoded in enumerate(result['decoded']):
        if decoded['success']:
            print(f"  二维码{i+1}: {decoded['data']}")
        else:
            print(f"  二维码{i+1}: 解码失败")

    return result


def process_camera(detector: QRCodeDetector, camera_id: int = 0):
    """使用摄像头实时检测"""
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    print("按 'q' 退出，按 's' 保存当前帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(img)

        # 可视化
        vis_img = visualize_result(img, result)

        # 显示
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('QR Code Detection', vis_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"qr_capture_{cv2.getTickCount()}.jpg"
            cv2.imwrite(filename, vis_img)
            print(f"Saved to {filename}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='QR Code Detection and Recognition')
    parser.add_argument('--model', type=str, default='models/qrcode_detect/best_detector.pth',
                       help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path, directory, or "camera"')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    args = parser.parse_args()

    # 创建检测器
    detector = QRCodeDetector(args.model, args.device)

    # 处理
    if args.input == 'camera':
        # 摄像头模式
        process_camera(detector)
    else:
        # 图像模式
        input_path = Path(args.input)

        # 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            # 单张图像
            output_path = output_dir / f"{input_path.stem}_result.jpg"
            process_image(detector, str(input_path), str(output_path))
        elif input_path.is_dir():
            # 目录中的图像
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(input_path.glob(ext)))

            print(f"\n处理 {len(image_files)} 张图像...")

            for img_path in image_files:
                output_path = output_dir / f"{img_path.stem}_result.jpg"
                process_image(detector, str(img_path), str(output_path))
        else:
            print(f"Error: Invalid input path {args.input}")


if __name__ == "__main__":
    main()