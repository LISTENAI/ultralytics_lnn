#!/usr/bin/env python3
"""
推理测试脚本 - 测试模型推理性能

支持模型格式:
- .pt: PyTorch 模型 (通过 ultralytics)
- .onnx: ONNX 模型

使用方法:
    # 查看帮助
    python tasks/run_inference.py --help

    # 基本推理测试
    python tasks/run_inference.py --model models/yolo26n.pt --source data/images

    # ONNX 模型推理
    python tasks/run_inference.py --model models/yolo26n.onnx --source data/images

    # 性能测试 (多次推理)
    python tasks/run_inference.py --model models/yolo26n.pt --benchmark --runs 100
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional
import glob as glob_module

# 项目根目录
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

import torch
import numpy as np


def get_model_type(model_path: str) -> str:
    """根据文件扩展名判断模型类型"""
    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt' or suffix == '.pth':
        return 'pytorch'
    elif suffix == '.onnx':
        return 'onnx'
    elif suffix == '.tflite':
        return 'tflite'
    else:
        return 'unknown'


def load_pytorch_model(model_path: str, device: str = 'cuda'):
    """加载 PyTorch 模型"""
    from ultralytics import YOLO

    print(f"📦 加载 PyTorch 模型: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    return model


def load_onnx_model(model_path: str, providers: List[str] = None):
    """加载 ONNX 模型"""
    import onnxruntime as ort

    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    print(f"📦 加载 ONNX 模型: {model_path}")
    session = ort.InferenceSession(model_path, providers=providers)

    # 获取输入输出信息
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"   输入: {[(inp.name, inp.shape) for inp in inputs]}")
    print(f"   输出: {[(out.name, out.shape) for out in outputs]}")

    return session


def get_image_files(source: str) -> List[str]:
    """获取图片文件列表"""
    source_path = Path(source)

    if source_path.is_file():
        return [str(source_path)]

    # 支持的格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    images = []

    for ext in extensions:
        images.extend(glob_module.glob(str(source_path / ext)))
        images.extend(glob_module.glob(str(source_path / ext.upper())))

    return sorted(images)


def run_pytorch_inference(model, source: str, imgsz: int = 640,
                          conf: float = 0.25, device: str = 'cuda'):
    """运行 PyTorch 模型推理"""
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )
    return results


def run_onnx_inference(session, images: List[str], imgsz: int = 640):
    """运行 ONNX 模型推理"""
    import onnxruntime as ort
    from PIL import Image

    # 预处理
    input_tensor = session.get_inputs()[0]
    input_name = input_tensor.name
    input_shape = input_tensor.shape

    # 获取实际输入尺寸
    if isinstance(input_shape[2], int):
        h, w = input_shape[2], input_shape[3]
    else:
        h, w = imgsz, imgsz

    results = []

    for img_path in images:
        # 加载并预处理图片
        img = Image.open(img_path).convert('RGB')
        img = img.resize((w, h))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)  # 添加 batch 维度

        # 推理
        outputs = session.run(None, {input_name: img_np})
        results.append(outputs)

    return results


def benchmark_pytorch(model, imgsz: int = 640, runs: int = 100,
                     warmup: int = 10, device: str = 'cuda'):
    """性能测试 PyTorch 模型"""
    print(f"\n⚡ PyTorch 性能测试 ({runs} 次推理)")
    print("=" * 50)

    # 预热
    print(f"预热 {warmup} 次...")
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.model(dummy_input) if hasattr(model, 'model') else model(dummy_input)

    # 正式测试
    print(f"开始测试...")
    torch.cuda.synchronize() if device.startswith('cuda') else None

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model.model(dummy_input) if hasattr(model, 'model') else model(dummy_input)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    # 统计
    times = np.array(times)
    print(f"\n结果 (ms):")
    print(f"  平均: {times.mean():.2f}")
    print(f"  标准差: {times.std():.2f}")
    print(f"  最小: {times.min():.2f}")
    print(f"  最大: {times.max():.2f}")
    print(f"  中位数: {np.median(times):.2f}")

    # FPS
    fps = 1000.0 / times.mean()
    print(f"\n  FPS: {fps:.1f}")

    return times


def benchmark_onnx(session, imgsz: int = 640, runs: int = 100, warmup: int = 10):
    """性能测试 ONNX 模型"""
    print(f"\n⚡ ONNX 性能测试 ({runs} 次推理)")
    print("=" * 50)

    input_tensor = session.get_inputs()[0]
    input_name = input_tensor.name

    # 创建 dummy 输入
    h, w = imgsz, imgsz
    dummy_input = np.random.randn(1, 3, h, w).astype(np.float32)

    # 预热
    print(f"预热 {warmup} 次...")
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # 正式测试
    print(f"开始测试...")
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # 统计
    times = np.array(times)
    print(f"\n结果 (ms):")
    print(f"  平均: {times.mean():.2f}")
    print(f"  标准差: {times.std():.2f}")
    print(f"  最小: {times.min():.2f}")
    print(f"  最大: {times.max():.2f}")

    fps = 1000.0 / times.mean()
    print(f"\n  FPS: {fps:.1f}")

    return times


def run_inference(args):
    """运行推理"""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return False

    model_type = get_model_type(model_path)
    print(f"\n📊 模型信息:")
    print(f"  路径: {model_path}")
    print(f"  类型: {model_type}")
    print(f"  设备: {args.device}")

    # 加载模型
    if model_type == 'pytorch':
        model = load_pytorch_model(model_path, args.device)
        model_loaded = model
    elif model_type == 'onnx':
        session = load_onnx_model(model_path)
        model_loaded = session
    else:
        print(f"❌ 不支持的模型类型: {model_type}")
        return False

    # 基准测试模式
    if args.benchmark:
        if model_type == 'pytorch':
            benchmark_pytorch(model, args.imgsz, args.runs, args.warmup, args.device)
        elif model_type == 'onnx':
            benchmark_onnx(session, args.imgsz, args.runs, args.warmup)
        return True

    # 推理模式
    if args.source:
        images = get_image_files(args.source)
        if not images:
            print(f"❌ 未找到图片: {args.source}")
            return False

        print(f"\n🖼️  找到 {len(images)} 张图片")

        if model_type == 'pytorch':
            results = run_pytorch_inference(model, args.source, args.imgsz,
                                           args.conf, args.device)
            print(f"\n✅ 推理完成，处理 {len(results)} 张图片")

            # 保存结果
            if args.save:
                for r in results:
                    if r.plot():
                        print(f"   保存: {r.path}")

        elif model_type == 'onnx':
            results = run_onnx_inference(session, images, args.imgsz)
            print(f"\n✅ 推理完成，处理 {len(results)} 张图片")

        return True

    print("❌ 请指定 --source 或 --benchmark")
    return False


def main():
    parser = argparse.ArgumentParser(
        description='推理测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 推理测试
  python tasks/run_inference.py --model models/yolo26n.pt --source val/images

  # ONNX 推理
  python tasks/run_inference.py --model models/yolo26n.onnx --source val/images

  # 性能基准测试
  python tasks/run_inference.py --model models/yolo26n.pt --benchmark --runs 100
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        help='模型路径 (.pt, .onnx)')

    parser.add_argument('--source', type=str,
                        help='图片或图片目录路径')

    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像尺寸')

    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备')

    parser.add_argument('--benchmark', action='store_true',
                        help='运行性能基准测试')

    parser.add_argument('--runs', type=int, default=100,
                        help='基准测试次数')

    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数')

    parser.add_argument('--save', action='store_true',
                        help='保存推理结果图片')

    args = parser.parse_args()

    # 设置 CUDA 设备
    if 'cuda' in args.device and torch.cuda.is_available():
        gpu_id = args.device.split(':')[1] if ':' in args.device else '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    success = run_inference(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()