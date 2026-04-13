# YOLO 量化训练部署项目模板

本项目是一个**通用的目标检测量化训练参考设计**，可快速适配不同检测任务。

## 项目目的

本工程作为**项目模板**，展示如何：
1. 使用 YOLO 进行目标检测训练
2. 集成 LISTENAI linger 量化训练
3. 使用 thinker 进行模型打包部署
4. 快速创建新的检测任务

## 快速开始

### 1. 环境准备

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 下载第三方依赖
python thirdparty/download_thirdparty.py
```

### 2. 创建新任务

参考 `tasks/cat_320X320_detect` 或 `tasks/dog_detection` 模板：

```bash
# 复制任务模板
cp -r tasks/cat_320X320_detect tasks/my_detection

# 修改配置文件
vim tasks/my_detection/cfg/train_config.yaml

# 修改数据集和模型路径
vim tasks/my_detection/cfg/datasets/my_data.yaml
vim tasks/my_detection/cfg/models/my_model.yaml
```

### 3. 训练模型

```bash
# 浮点训练
python tasks/my_detection/train_my_task.py

# 量化训练
python tasks/my_detection/train_my_task.py --quant
```

### 4. 导出与打包

```bash
# 导出 ONNX
python tasks/my_detection/test_my_task.py --export-onnx

# 优化 ONNX
python tasks/pack_model.py --model best.onnx --optimize-onnx

# 打包模型
python tasks/pack_model.py --model best_int8.onnx --platform venusA
```

## 目录结构

```
.
├── tasks/                    # 任务目录
│   ├── pack_model.py         # 模型打包工具
│   ├── run_inference.py      # 推理测试工具
│   ├── cat_320X320_detect/   # 猫脸检测任务 (示例)
│   │   ├── train_*.py
│   │   ├── test_*.py
│   │   ├── prepare_dataset.py
│   │   └── cfg/
│   └── dog_detection/        # 狗脸检测任务 (示例)
├── thirdparty/               # 第三方依赖
│   ├── download_thirdparty.py
│   ├── linger/               # 量化训练框架
│   └── thinker/              # 模型打包框架
├── tools/                    # 辅助工具
│   ├── analyze_onnx.py       # ONNX 模型分析
│   └── export_onnx.py        # ONNX 导出
├── docs/                     # 项目文档
└── runs/                     # 训练输出
```

## 可用任务

| 任务 | 说明 | 数据集 |
|------|------|--------|
| cat_320X320_detect | 猫脸检测 | Oxford-IIIT Pet (猫) |
| dog_detection | 狗脸检测 | Oxford-IIIT Pet (狗) |

## 工具使用

### ONNX 模型分析

```bash
python tools/analyze_onnx.py model.onnx --profile
```

### 推理测试

```bash
python tasks/run_inference.py --model model.pt --source images/
```

## 扩展新任务

1. 在 `tasks/` 下创建新目录
2. 添加 `train_<task>.py` 训练脚本
3. 添加 `test_<task>.py` 测试脚本
4. 在 `cfg/` 下添加数据集和模型配置

详见 [.claude/skills/new_task/README.md](.claude/skills/new_task/README.md)

## 相关文档

- [LISTENAI 量化工具链](../.claude/skills/listenai_quant/README.md)
- [新建任务指南](../.claude/skills/new_task/README.md)
- [第三方依赖安装](../thirdparty/README.md)