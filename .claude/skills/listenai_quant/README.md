# LISTENAI 量化训练与部署工具链

## 概述

LISTENAI 工具链是专为聆思 AIOT 芯片设计的端到端深度学习部署解决方案，包含两大核心组件：

| 组件 | 功能 | 链接 |
|------|------|------|
| **linger** | 基于 PyTorch 的 QAT 量化训练 | [GitHub](https://github.com/LISTENAI/linger) |
| **thinker** | 轻量级推理框架 + 模型打包工具 | [GitHub](https://github.com/LISTENAI/thinker) |

> **项目定位**：本工程提供一套通用的目标检测量化训练参考设计，可快速适配不同检测任务（如猫脸、狗脸、车辆等）。

---

## 安装第三方依赖

### 自动下载（推荐）

```bash
# 使用 Python 脚本
python thirdparty/download_thirdparty.py

# 或使用 Bash 脚本
bash thirdparty/download_thirdparty.sh
```

### 手动克隆

```bash
# 克隆 linger
git clone --depth 1 https://github.com/LISTENAI/linger.git thirdparty/linger

# 克隆 thinker
git clone --depth 1 https://github.com/LISTENAI/thinker.git thirdparty/thinker
```

### 安装依赖

```bash
# 安装 linger
cd thirdparty/linger && pip install -e .

# 安装 thinker
cd thirdparty/thinker && pip install -e .
cd thirdparty/thinker/tools && pip install -e .
```

---

## 支持的芯片平台

| 平台 | 说明 |
|------|------|
| **venusA** | 聆思 VenusA 芯片（推荐） |
| **venus** | 聆思 Venus 芯片 |
| **arcs** | 聆思 ARCS 芯片 |

---

## 目录结构设计

```
tasks/                    # 任务目录
├── <task_name>/           # 任务目录（如 cat_320X320_detect）
│   ├── prepare_dataset.py # 数据集准备脚本
│   ├── train_<task>.py    # 训练脚本
│   ├── test_<task>.py     # 测试脚本
│   └── cfg/                # 任务配置目录
│       ├── train_config.yaml
│       ├── datasets/
│       ├── models/
│       ├── quant_config.yaml
│       └── pack_config.yaml
├── pack_model.py          # 模型打包工具
└── run_inference.py        # 推理测试工具

thirdparty/                 # 第三方依赖（需下载）
```

---

## 快速开始

### 1. 下载依赖

```bash
python thirdparty/download_thirdparty.py
```

### 2. 配置文件方式

修改 `tasks/<task>/cfg/train_config.yaml`：

```yaml
epochs: 50
imgsz: 320
batch: 32
device: 2
lr: 0.01
```

### 3. 运行训练

```bash
# 浮点训练
python tasks/cat_320X320_detect/train_cat_face_320.py

# 量化训练
python tasks/cat_320X320_detect/train_cat_face_320.py --quant
```

---

## 量化训练流程

```
浮点约束训练 → 量化微调 → ONNX导出 → 模型打包
```

---

## 常用命令速查

| 任务 | 命令 |
|------|------|
| 下载依赖 | `python thirdparty/download_thirdparty.py` |
| 浮点训练 | `python tasks/<task_dir>/train_<task>.py` |
| 量化约束训练 | `python tasks/<task_dir>/train_<task>.py --quant --quant-stage constrain` |
| 量化训练 | `python tasks/<task_dir>/train_<task>.py --quant --quant-stage quant` |
| 模型测试 | `python tasks/<task_dir>/test_<task>.py` |
| ONNX优化 | `python tasks/pack_model.py --model model.onnx --optimize-onnx` |
| 模型打包 | `python tasks/pack_model.py --model model.onnx --platform venusA` |

---

## 相关文档

- [NPU 算子支持与限制说明](./npu_limits.md) - **重要：设计模型前必读**
- [linger 量化训练文档](https://github.com/LISTENAI/linger)
- [thinker 推理框架文档](https://github.com/LISTENAI/thinker)
- [第三方依赖安装](thirdparty/README.md)