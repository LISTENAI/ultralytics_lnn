#!/usr/bin/env python3
"""
Cat Face Detection Test Script (320x320)
测试训练好的模型在验证集上的性能
"""

import os
import sys
import argparse
from pathlib import Path

# 脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()
TASK_NAME = SCRIPT_DIR.name
CFG_DIR = SCRIPT_DIR / 'cfg'

# 项目根目录
ULTRALYTICS_PATH = SCRIPT_DIR.parent.parent

from ultralytics import YOLO


def main(args):
    # 模型路径
    model_path = args.model

    # 数据集配置
    data_yaml = str(CFG_DIR / 'datasets' / 'cat_face_320.yaml')

    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Please check the training output directory.")
        return

    # 加载模型
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # 验证集配置
    imgsz = args.imgsz
    batch = args.batch

    print(f"\nRunning validation on {args.split} set...")
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        split=args.split,
        verbose=True,
    )

    # 打印关键指标
    print("\n" + "=" * 50)
    print("Validation Results Summary:")
    print("=" * 50)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print("=" * 50)

    # 推理测试
    if args.predict:
        print("\nRunning inference samples...")
        val_images_dir = args.source

        if os.path.exists(val_images_dir):
            import glob
            image_files = glob.glob(os.path.join(val_images_dir, '*.jpg'))[:5]

            for img_path in image_files:
                model.predict(
                    source=img_path,
                    imgsz=imgsz,
                    conf=0.25,
                    iou=0.45,
                    device=args.device,
                    save=True,
                    project=os.path.join(ULTRALYTICS_PATH, 'runs/detect'),
                    name='test_output',
                )
                print(f"Processed: {img_path}")

    print("\nTest completed!")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cat Face Detection Test')
    parser.add_argument('--model', type=str,
                        default='/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face/runs/detect/cat_face_yolo26n_train_320/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--imgsz', type=int, default=320, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=int, default=2, help='GPU设备')
    parser.add_argument('--split', type=str, default='val', help='数据集划分')
    parser.add_argument('--predict', action='store_true', help='运行推理测试')
    parser.add_argument('--source', type=str,
                        default='/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face/val/images',
                        help='推理图片目录')

    args = parser.parse_args()

    # 设置 CUDA 设备
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.device))

    main(args)