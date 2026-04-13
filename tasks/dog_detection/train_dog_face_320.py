#!/usr/bin/env python3
"""
Dog Face Detection Training Script (320x320)
使用320x320输入尺寸进行训练，速度更快、显存占用更少

使用方法（简化版）:
    python train_dog_face_320.py                    # 使用默认配置
    python train_dog_face_320.py --epochs 10        # 覆盖参数
    python train_dog_face_320.py --quant             # 启用量化训练

配置文件: cfg/train_config.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# 脚本所在目录 (任务根目录)
SCRIPT_DIR = Path(__file__).parent.resolve()
TASK_NAME = SCRIPT_DIR.name
CFG_DIR = SCRIPT_DIR / 'cfg'

# 项目根目录 (从 tasks/<task>/ 向上2级)
ULTRALYTICS_PATH = SCRIPT_DIR.parent.parent

# 添加 linger 路径
LINGER_PATH = ULTRALYTICS_PATH / 'thirdparty' / 'linger'
if LINGER_PATH.exists() and str(LINGER_PATH) not in sys.path:
    sys.path.insert(0, str(LINGER_PATH))

# 清理 linger 模块缓存
if 'linger' in sys.modules:
    del sys.modules['linger']
for mod in list(sys.modules.keys()):
    if mod.startswith('linger'):
        del sys.modules[mod]

from ultralytics import YOLO

# linger 可用性检查
LINGER_AVAILABLE = False
try:
    import linger
    from linger import init, constrain
    LINGER_AVAILABLE = True
except ImportError:
    pass


def load_config(config_path=None):
    """加载训练配置"""
    if config_path is None:
        config_path = CFG_DIR / 'train_config.yaml'

    if not os.path.exists(config_path):
        print(f"⚠️ 配置文件不存在: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"📄 加载配置: {config_path}")
    return config


def get_config_value(args, config, key, default=None):
    """获取配置值：命令行参数 > 配置文件 > 默认值"""
    cmd_value = getattr(args, key, None)
    if cmd_value is not None and (not isinstance(cmd_value, bool) or cmd_value):
        return cmd_value
    return config.get(key, default)


def add_quantization(model, config_file, stage='quant'):
    """在模型中添加量化"""
    if not LINGER_AVAILABLE:
        print("⚠️ linger 不可用，跳过量量化")
        return model

    if not os.path.exists(config_file):
        print(f"⚠️ 配置文件不存在: {config_file}")
        config_file = None

    if stage == 'constrain':
        print(f"🔧 添加浮点约束训练...")
        return constrain(model, config_file=config_file)
    else:
        print(f"🔧 添加量化训练...")
        return init(
            model,
            config_file=config_file,
            disable_submodel=('*9.m',)
        )


def export_quant_onnx(model, output_path, input_shape=(3, 320, 320), opset=12):
    """导出量化 ONNX 模型"""
    if not LINGER_AVAILABLE:
        print("⚠️ linger 不可用，无法导出")
        return

    import torch
    model.eval()
    dummy_input = torch.randn(1, *input_shape)

    with torch.no_grad():
        linger.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    print(f"✅ 量化模型已导出: {output_path}")


def main(args):
    # 加载配置文件
    config = load_config(args.config)

    # 获取参数（命令行 > 配置 > 默认值）
    epochs = get_config_value(args, config, 'epochs', 50)
    imgsz = get_config_value(args, config, 'imgsz', 320)
    batch = get_config_value(args, config, 'batch', 32)
    device = get_config_value(args, config, 'device', 2)
    lr = get_config_value(args, config, 'lr', 0.01)
    workers = get_config_value(args, config, 'workers', 8)
    patience = get_config_value(args, config, 'patience', 100)
    project = get_config_value(args, config, 'project', './runs/detect')
    name = get_config_value(args, config, 'name', 'dog_face_yolo26n_train_320')
    quant = get_config_value(args, config, 'quant', False)
    quant_stage = get_config_value(args, config, 'quant_stage', 'both')
    weight_decay = get_config_value(args, config, 'weight_decay', 0.0005)
    momentum = get_config_value(args, config, 'momentum', 0.937)

    # 数据集和模型配置
    dataset_file = get_config_value(args, config, 'dataset', 'dog.yaml')
    model_file = get_config_value(args, config, 'model', 'yolo26n_dog.yaml')

    data_yaml = str(CFG_DIR / 'datasets' / dataset_file)
    model_yaml = str(CFG_DIR / 'models' / model_file)
    quant_config = str(CFG_DIR / 'quant_config.yaml')

    print(f"\n{'=' * 60}")
    print(f"📋 训练配置:")
    print(f"  epochs: {epochs}, imgsz: {imgsz}, batch: {batch}")
    print(f"  device: {device}, lr: {lr}")
    print(f"  quant: {quant}, quant_stage: {quant_stage}")
    print(f"  data: {data_yaml}")
    print(f"  model: {model_yaml}")
    print(f"{'=' * 60}\n")

    # 加载模型
    print(f"Loading model from: {model_yaml}")
    model = YOLO(model_yaml)

    # 量化训练
    if quant:
        print(f"\n{'=' * 60}")
        print(f"🔧 量化训练模式: {quant_stage}")
        print(f"{'=' * 60}")

        # 阶段1：浮点约束训练
        if quant_stage in ('both', 'constrain'):
            print("\n📌 阶段1: 浮点约束训练")
            model.model = add_quantization(model.model, quant_config, 'constrain')

            results1 = model.train(
                data=data_yaml,
                epochs=epochs // 2 if quant_stage == 'both' else epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=project,
                name=name + '_constrain',
                lr0=lr,
                lrf=lr,
                optimizer='SGD',
                momentum=momentum,
                weight_decay=weight_decay,
                workers=workers,
                verbose=True,
                seed=0,
                deterministic=True,
                patience=patience,
            )
            print("✅ 浮点约束训练完成")

            if quant_stage == 'both':
                model = YOLO(model_yaml)

        # 阶段2：量化训练
        if quant_stage in ('both', 'quant'):
            print("\n📌 阶段2: 量化训练")
            model.model = add_quantization(model.model, quant_config, 'quant')

            results2 = model.train(
                data=data_yaml,
                epochs=epochs // 2 if quant_stage == 'both' else epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=project,
                name=name + '_quant',
                lr0=lr / 10,
                lrf=lr / 10,
                optimizer='SGD',
                momentum=momentum,
                weight_decay=weight_decay,
                workers=workers,
                verbose=True,
                seed=0,
                deterministic=True,
                patience=patience,
            )

            # 导出量化 ONNX
            output_dir = results2.save_dir
            onnx_path = os.path.join(output_dir, 'model_quant.onnx')
            export_quant_onnx(model.model, onnx_path, input_shape=(3, imgsz, imgsz))

            print(f"\n✅ 量化训练完��!")
            print(f"  模型: {results2.save_dir}")
            print(f"  ONNX: {onnx_path}")

            return results2

        return results1 if quant_stage == 'constrain' else None

    # 普通浮点训练
    print(f"\n{'=' * 60}")
    print("📌 标准浮点训练模式")
    print(f"{'=' * 60}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        lr0=lr,
        lrf=lr,
        optimizer='SGD',
        momentum=momentum,
        weight_decay=weight_decay,
        workers=workers,
        verbose=True,
        seed=0,
        deterministic=True,
        patience=patience,
    )

    print("\nTraining completed!")
    print(f"Results saved to: {results.save_dir}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dog Face Detection Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_dog_face_320.py                    # 使用默认配置
  python train_dog_face_320.py --epochs 10        # 覆盖训练轮数
  python train_dog_face_320.py --device 0         # 使用GPU 0
  python train_dog_face_320.py --quant            # 启用量化训练
  python train_dog_face_320.py --quant --quant-stage constrain  # 仅约束训练
        """
    )
    # 配置文件（可选）
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (默认: cfg/train_config.yaml)')

    # 常用参数（可覆盖配置文件）
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch', type=int, help='批次大小')
    parser.add_argument('--device', type=int, help='GPU设备ID')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--imgsz', type=int, help='图像尺寸')

    # 量化参数
    parser.add_argument('--quant', action='store_true', help='启用量化训练')
    parser.add_argument('--quant-stage', type=str,
                        choices=['both', 'constrain', 'quant'],
                        help='量化阶段')

    # 输出配置
    parser.add_argument('--name', type=str, help='实验名称')

    args = parser.parse_args()

    # 设置 CUDA 设备
    if args.device is not None:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.device))

    main(args)