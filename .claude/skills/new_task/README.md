# 新建任务指南

本指南帮助您快速创建新的检测、识别或分割任务。

---

## 目录规范（重要）

项目采用 **代码与运行时产物分离** 的目录结构，**task 目录下只包含源代码、配置文件和文档**：

```
项目根目录/
├── tasks/                    # 任务代码（只包含 .py, .yaml, .md 等源码）
│   └── <task_name>/
│       ├── scripts/          # 脚本目录（可选）
│       │   ├── train_xxx.py
│       │   └── inference_xxx.py
│       └── cfg/              # 配置文件目录
│           ├── train_config.yaml
│           └── datasets/
├── dataset/                  # 数据集目录
│   └── <task_name>/
├── models/                   # 模型文件目录
│   └── <task_name>/
│       └── best_model.pth
├── runs/                     # 训练输出目录
│   └── <task_name>/
│       ├── best_model.pth
│       └── checkpoint_*.pth
├── output/                   # 推理结果目录
│   └── <task_name>/
└── docs/                     # 项目文档
```

### 为什么这样做？
- **代码版本控制**：task 目录只包含源码，可以被 git 追踪
- **运行时产物隔离**：模型、训练日志等存放在根目录，不会被误提交
- **多任务共享**：不同任务可以共享同一个 dataset/models/runs 目录

---

## 快速开始

推荐使用 `cat_320X320_detect` 作为模板进行修改：

```bash
# 1. 复制模板目录
cp -r tasks/cat_320X320_detect tasks/<your_task_name>

# 2. 重命名文件
mv tasks/<your_task_name>/train_cat_face_320.py tasks/<your_task_name>/train_<your_task>.py
mv tasks/<your_task_name>/test_cat_face_320.py tasks/<your_task_name>/test_<your_task>.py
```

---

## 目录结构

```
tasks/<task_name>/
├── prepare_dataset.py  # 数据集准备脚本
├── train_<task>.py     # 训练脚本
├── test_<task>.py      # 测试脚本
└── cfg/                 # 配置文件目录
    ├── train_config.yaml      # 训练配置
    ├── quant_config.yaml      # 量化配置
    ├── pack_config.yaml       # 打包配置
    ├── datasets/              # 数据集配置
    │   └── cat_face_320.yaml
    └── models/                # 模型配置
        └── yolo26n_cat.yaml
```

---

## 配置文件说明

### 1. train_config.yaml

```yaml
# 训练参数
epochs: 10          # 训练轮数
imgsz: 320         # 输入图像尺寸
batch: 32          # 批次大小
device: 2          # GPU设备ID

# 优化器参数
lr: 0.01           # 初始学习率
weight_decay: 0.0005
momentum: 0.937

# 训练选项
workers: 8         # 数据加载线程数
patience: 100      # 早停耐心值

# 输出设置
project: ./runs/detect
name: cat_face_yolo26n_train_320

# 量化训练选项
quant: false        # 是否启用量化训练
quant_stage: quant  # 量化阶段: both/constrain/quant

# 数据集配置 (相对于 cfg 目录)
dataset: cat_face_320.yaml
model: yolo26n_cat.yaml
```

### 2. 数据集配置 (datasets/xxx.yaml)

```yaml
# 训练集
train: ../dataset/train/images

# 验证集
val: ../dataset/val/images

# 类别数量
nc: 1

# 类别名称
names:
  0: cat
```

### 3. 模型配置 (models/xxx.yaml)

基于 YOLO26n 修改输出层：

```yaml
# 类别数量 (根据任务修改)
nc: 1

# 类别名称
names:
  0: cat
```

---

## 训练脚本修改

### train_<task>.py 关键修改点

1. **配置文件路径**：
```python
# 修改任务名称
CFG_DIR = Path(__file__).parent / "cfg"
CONFIG_FILE = CFG_DIR / "train_config.yaml"
```

2. **数据集配置**：
```yaml
# cfg/<task_name>.yaml 中修改数据集路径
train: /path/to/your/train/images
val: /path/to/your/val/images
```

3. **模型配置**：
```yaml
# cfg/models/<model_name>.yaml 中修改 nc (类别数)
nc: 1  # 你的类别数量
```

### 命令行参数默认值规范（重要）

为了保持一致性，训练和推理脚本的命令行参数应使用**相对于项目根目录的路径**：

```python
import argparse
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

def main():
    parser = argparse.ArgumentParser(description='Train Task')
    # 数据集目录：使用相对路径，默认为 dataset/<task_name>
    parser.add_argument('--data-dir', type=str,
                       default='dataset/my_task',
                       help='数据集目录（相对于项目根目录）')

    # 模型输出目录：使用 runs/<task_name>
    parser.add_argument('--output', type=str,
                       default='runs/my_task',
                       help='输出目录（相对于项目根目录）')

    # 模型文件路径：使用 models/<task_name>
    parser.add_argument('--model', type=str,
                       default='models/my_task/best_model.pth',
                       help='模型文件路径')

    # 推理输出目录：使用 output/<task_name>
    parser.add_argument('--output-dir', type=str,
                       default='output/my_task',
                       help='输出目录')

    args = parser.parse_args()

    # 转换为绝对路径（可选）
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output
```

### 常见错误避免

| 错误做法 | 正确做法 |
|---------|---------|
| `default='/CodeRepo/.../dataset/xxx'` | `default='dataset/xxx'` |
| `default='runs/xxx'`（与task重名） | `default='runs/my_task'` |
| 模型保存到 `tasks/xxx/runs/` | 模型保存到 `runs/xxx/` 或 `models/xxx/` |
| 输出目录放在 task 目录下 | 输出目录放在根目录的 output/ 下 |

---

## 不同任务类型配置

### 目标检测 (Detection)

```yaml
# train_config.yaml
task: detect
mode: train
model: yolo26n.yaml
```

### 图像分割 (Segmentation)

```yaml
# train_config.yaml
task: segment
mode: train
model: yolo26n-seg.yaml
```

### 图像分类/识别 (Classification)

```yaml
# train_config.yaml
task: classify
mode: train
model: yolo26n-cls.yaml
```

---

## VSCode 调试配置

在 `.vscode/launch.json` 中添加新任务的调试配置：

```json
{
    "name": "Train <Task Name>",
    "type": "debugpy",
    "request": "launch",
    "module": "ultralytics",
    "args": [
        "train",
        "data=${workspaceFolder}/tasks/<task_name>/cfg/datasets/<dataset>.yaml",
        "model=${workspaceFolder}/tasks/<task_name>/cfg/models/<model>.yaml",
        "epochs=1",
        "imgsz=320",
        "device=0"
    ],
    "console": "integratedTerminal"
}
```

---

## 量化训练 (可选)

如需启用量化训练：

```bash
# 约束训练阶段
python tasks/<task_name>/train_<task>.py --quant --quant-stage constrain

# 量化训练阶段
python tasks/<task_name>/train_<task>.py --quant --quant-stage quant

# 一键量化训练
python tasks/<task_name>/train_<task>.py --quant --quant-stage both
```

---

## 完整流程

1. **创建任务目录**
   ```bash
   mkdir -p tasks/<task_name>/cfg/datasets
   mkdir -p tasks/<task_name>/cfg/models
   ```

2. **创建运行时目录**（在项目根目录）
   ```bash
   mkdir -p dataset/<task_name>
   mkdir -p models/<task_name>
   mkdir -p runs/<task_name>
   mkdir -p output/<task_name>
   ```

3. **配置文件**
   - 复制并修改 `train_config.yaml`
   - 创建/修改数据集配置文件（使用相对路径 `dataset/<task_name>/`）
   - 创建/修改模型配置文件

4. **脚本文件**
   - 复制并修改 `train_<task>.py`
   - 复制并修改 `test_<task>.py`
   - **注意**：脚本中的默认路径使用相对于项目根目录的路径

5. **测试运行**
   ```bash
   python tasks/<task_name>/train_<task>.py --epochs 1
   ```

6. **导出部署**
   ```bash
   # 导出 ONNX（模型会保存到 runs/<task_name>/）
   python tasks/<task_name>/test_<task>.py --export-onnx

   # 模型打包（使用 runs/<task_name>/ 下的模型）
   python tasks/pack_model.py --model runs/<task_name>/best.onnx --platform venusA
   ```

---

## 相关文档

- [LISTENAI 量化工具链](../listenai_quant/README.md)
- [NPU 算子支持与限制说明](../listenai_quant/npu_limits.md) - **模型设计前必读**
- [训练脚本示例](../cat_320X320_detect/train_cat_face_320.py)
- [项目 README](../../README.md)