#!/usr/bin/env python3
"""
Cat Face Detection - Export to ONNX
将训练好的模型导出为 ONNX 格式
"""

import os

from ultralytics import YOLO

# 项目根目录
ULTRALYTICS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_onnx(
    model_path: str = None,
    img_size: int = 640,
    simplify: bool = True,
    opset: int = 12
):
    """
    导出模型为 ONNX 格式

    Args:
        model_path: 模型路径，默认使用最佳权重
        img_size: 输入图像大小
        simplify: 是否简化 ONNX 模型
        opset: ONNX opset 版本
    """
    # 默认模型路径
    if model_path is None:
        model_path = os.path.join(
            'yolo26n_320.pt'
        )

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # 导出 ONNX
    print(f"Exporting to ONNX (simplify={simplify}, opset={opset}, img_size={img_size})...")
    export_path = model.export(
        format='onnx',
        imgsz=img_size,
        simplify=simplify,
        opset=opset
    )

    print(f"\nExport completed!")
    print(f"ONNX model saved to: {export_path}")

    return export_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Export YOLO model to ONNX')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径 (默认: best.pt)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像大小 (默认: 640)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='不简化 ONNX 模型')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset 版本 (默认: 12)')

    args = parser.parse_args()

    export_onnx(
        model_path=args.model,
        img_size=args.img_size,
        simplify=not args.no_simplify,
        opset=args.opset
    )


if __name__ == '__main__':
    # 设置 CUDA 设备
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

    main()
