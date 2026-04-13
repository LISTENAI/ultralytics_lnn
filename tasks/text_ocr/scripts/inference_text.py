#!/usr/bin/env python3
"""
文本检测-分割-OCR完整推理脚本
整合三个阶段的模型进行端到端文本识别
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from text_models import TextDetector, TextSegmenter, CRNN


class TextRecognitionPipeline:
    """文本识别完整流水线"""
    def __init__(self, detector_path: str, segmenter_path: str, ocr_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 加载模型
        print("Loading models...")

        # 文本检测器
        self.detector = TextDetector().to(self.device)
        if detector_path and Path(detector_path).exists():
            ckpt = torch.load(detector_path, map_location=self.device)
            self.detector.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"   ✅ Detector loaded: {detector_path}")
        self.detector.eval()

        # 文本分割器
        self.segmenter = TextSegmenter().to(self.device)
        if segmenter_path and Path(segmenter_path).exists():
            ckpt = torch.load(segmenter_path, map_location=self.device)
            self.segmenter.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"   ✅ Segmenter loaded: {segmenter_path}")
        self.segmenter.eval()

        # OCR
        self.ocr = CRNN(num_classes=6826).to(self.device)
        if ocr_path and Path(ocr_path).exists():
            ckpt = torch.load(ocr_path, map_location=self.device)
            self.ocr.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print(f"   ✅ OCR loaded: {ocr_path}")
        self.ocr.eval()

        # 参数量统计
        print("\n" + "=" * 40)
        print("Model Parameters")
        print("=" * 40)
        det_params = self.detector.count_parameters()
        seg_params = self.segmenter.count_parameters()
        ocr_params = self.ocr.count_parameters()
        print(f"   Detector: {det_params:,} ({det_params/1e6:.2f}M)")
        print(f"   Segmenter: {seg_params:,} ({seg_params/1e6:.2f}M)")
        print(f"   OCR: {ocr_params:,} ({ocr_params/1e6:.2f}M)")
        print(f"   Total: {det_params + seg_params + ocr_params:,} ({(det_params + seg_params + ocr_params)/1e6:.2f}M)")

        # 检查是否满足要求
        max_params = 4_000_000
        if det_params > max_params:
            print(f"⚠️ Detector exceeds {max_params/1e6:.0f}M!")
        if seg_params > max_params:
            print(f"⚠️ Segmenter exceeds {max_params/1e6:.0f}M!")
        if ocr_params > max_params:
            print(f"⚠️ OCR exceeds {max_params/1e6:.0f}M!")

    def preprocess(self, img: np.ndarray, target_size: int = 640) -> torch.Tensor:
        """预处理图像"""
        h, w = img.shape[:2]

        # 保持宽高比缩放
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h))

        # 创建画布并居中
        img_padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        img_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

        # 归一化并转换为tensor
        img_tensor = img_padded.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)

        return img_tensor.to(self.device)

    def detect_text(self, img_tensor: torch.Tensor) -> dict:
        """文本检测"""
        with torch.no_grad():
            output = self.detector(img_tensor)
        return output

    def segment_text(self, img_tensor: torch.Tensor) -> dict:
        """文本分割"""
        with torch.no_grad():
            output = self.segmenter(img_tensor)
        return output

    def recognize_text(self, text_regions: list) -> list:
        """OCR识别"""
        results = []

        for region in text_regions:
            if region['crop'] is None:
                results.append({**region, 'text': '', 'confidence': 0.0})
                continue

            # 预处理文本区域
            crop = region['crop']

            # 调整大小到OCR输入尺寸
            crop = cv2.resize(crop, (128, 32))

            # 灰度化
            if len(crop.shape) == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # 归一化
            crop = crop.astype(np.float32) / 255.0
            crop = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(self.device)

            # OCR识别
            with torch.no_grad():
                output = self.ocr(crop)
                output = output.softmax(2)

            # 解码
            text = self.decode_ctc(output)

            results.append({**region, 'text': text, 'confidence': float(output.max())})

        return results

    def decode_ctc(self, output: torch.Tensor) -> str:
        """CTC解码"""
        # 简化版贪婪解码
        pred = output.squeeze(1).cpu().numpy()
        result = []
        prev = -1

        for p in pred:
            idx = np.argmax(p)
            if idx != prev and idx < len(FULL_CHAR_SET):
                result.append(FULL_CHAR_SET[idx])
            prev = idx

        return ''.join(result)

    def process_image(self, image_path: str, output_path: str = None, save_visualization: bool = True):
        """处理单张图像"""
        print(f"\nProcessing: {image_path}")

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ❌ Failed to read image: {image_path}")
            return None

        original_h, original_w = img.shape[:2]
        print(f"   Image size: {original_w}x{original_h}")

        # 预处理
        img_tensor = self.preprocess(img, target_size=640)

        # 文本检测
        print("   Stage 1: Text Detection...")
        det_output = self.detect_text(img_tensor)
        prob_map = det_output['prob_map'].cpu().squeeze().numpy()
        binary_map = (prob_map > 0.5).astype(np.uint8) * 255

        # 找文本区域轮廓
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 文本分割
        print("   Stage 2: Text Segmentation...")
        seg_output = self.segment_text(img_tensor)
        seg_mask = seg_output['seg_mask'].cpu().squeeze().numpy()

        # 处理每个文本区域
        print("   Stage 3: OCR Recognition...")
        text_regions = []

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤太小的区域
            if w < 10 or h < 10:
                continue

            # 提取文本区域 (反归一化)
            scale_x = original_w / 640
            scale_y = original_h / 640

            rx = int(x * scale_x)
            ry = int(y * scale_y)
            rw = int(w * scale_x)
            rh = int(h * scale_y)

            # 裁剪文本区域
            crop = img[max(0, ry):min(original_h, ry+rh), max(0, rx):min(original_w, rx+rw)]

            text_regions.append({
                'index': len(text_regions),
                'bbox': [rx, ry, rx+rw, ry+rh],
                'crop': crop if crop.size > 0 else None
            })

        # OCR识别
        results = self.recognize_text(text_regions)

        # 打印结果
        print("\n" + "=" * 40)
        print("Recognition Results")
        print("=" * 40)

        for r in results:
            print(f"   [{r['index']}] {r['text']} (bbox: {r['bbox']})")

        # 可视化
        if save_visualization:
            vis_img = img.copy()

            # 绘制检测框
            for r in results:
                bbox = r['bbox']
                cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                # 绘制识别文本
                if r.get('text'):
                    cv2.putText(vis_img, r['text'], (bbox[0], bbox[1]-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 保存可视化结果
            if output_path:
                output_vis = str(Path(output_path).with_suffix('')) + '_vis.jpg'
                cv2.imwrite(output_vis, vis_img)
                print(f"\n   ✅ Visualization saved to: {output_vis}")

        # 保存JSON结果
        if output_path:
            output_json = str(Path(output_path).with_suffix('')) + '.json'
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump({
                    'image_path': image_path,
                    'image_size': [original_w, original_h],
                    'results': results
                }, f, ensure_ascii=False, indent=2)
            print(f"   ✅ Results saved to: {output_json}")

        return results


def main(args):
    # 创建推理pipeline
    pipeline = TextRecognitionPipeline(
        detector_path=args.detector,
        segmenter_path=args.segmenter,
        ocr_path=args.ocr,
        device=args.device
    )

    # 处理图像/目录
    input_path = Path(args.input)

    if input_path.is_file():
        # 单张图像
        pipeline.process_image(str(input_path), args.output)

    elif input_path.is_dir():
        # 目录中的所有图像
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_files = []
        for ext in img_extensions:
            img_files.extend(input_path.glob(f'*{ext}'))
            img_files.extend(input_path.glob(f'*{ext.upper()}'))

        print(f"\nFound {len(img_files)} images")

        for img_path in tqdm(img_files, desc='Processing images'):
            output_path = args.output / f"{img_path.stem}_result" if args.output else None
            pipeline.process_image(
                str(img_path),
                str(output_path) if output_path else None
            )

    print("\n✅ All done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Recognition Inference')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path')
    parser.add_argument('--detector', '-d', type=str,
                        default='runs/text_ocr/detector/best_detector.pth',
                        help='Detector model path')
    parser.add_argument('--segmenter', '-s', type=str,
                        default='runs/text_ocr/segmenter/best_segmenter.pth',
                        help='Segmenter model path')
    parser.add_argument('--ocr', '-r', type=str,
                        default='runs/text_ocr/ocr/best_ocr.pth',
                        help='OCR model path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()
    main(args)