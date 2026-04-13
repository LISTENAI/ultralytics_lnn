# YOLO 量化训练部署项目

基于 ultralytics YOLO 的目标检测量化训练项目，支持 LISTENAI 工具链（linger 量化训练 + thinker 推理打包）。

## 项目定位

本工程提供一套**通用的目标检测量化训练参考设计**，可快速适配不同检测任务：

- 动物检测（猫脸、狗脸等）
- 车辆检测
- 工业检测
- 自定义目标检测

---

## 目录结构

```
.
├── .vscode/                  # VSCode 调试配置
├── .claude/                  # Claude Code 配置
├── tasks/                    # 任务目录
│   ├── pack_model.py         # 模型打包工具
│   ├── run_inference.py      # 推理测试工具
│   └── <task_name>/          # 任务目录（参考: cat_320X320_detect）
│       ├── prepare_dataset.py # 数据集准备脚本
│       ├── train_<task>.py   # 训练脚本
│       ├── test_<task>.py    # 测试脚本
│       └── cfg/              # 任务配置目录
│           ├── train_config.yaml
│           ├── datasets/
│           ├── models/
│           ├── quant_config.yaml
│           └── pack_config.yaml
├── thirdparty/               # 第三方依赖（需下载）
│   ├── download_thirdparty.py # 第三方依赖下载脚本
│   ├── download_thirdparty.sh # 第三方依赖下载脚本 (Bash)
│   ├── linger/               # 量化训练框架
│   └── thinker/              # 模型打包框架
├── models/                   # 模型输出目录
├── tools/                    # 辅助工具
└── docs/                     # 项目文档 (模板使用指南)
```

---

## 快速开始

### 1. 环境准备

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 下载第三方依赖（linger + thinker）
python thirdparty/download_thirdparty.py
```

### 2. 训练模型

```bash
# 使用配置文件训练
python tasks/cat_320X320_detect/train_cat_face_320.py

# 命令行覆盖参数
python tasks/cat_320X320_detect/train_cat_face_320.py --epochs 10

# 启用量化训练
python tasks/cat_320X320_detect/train_cat_face_320.py --quant
```

---

## 第三方依赖

| 依赖 | 说明 | 链接 |
|------|------|------|
| **linger** | 量化训练框架 | [GitHub](https://github.com/LISTENAI/linger) |
| **thinker** | 模型打包框架 | [GitHub](https://github.com/LISTENAI/thinker) |

下载脚本会自动克隆这两个仓库到 `thirdparty/` 目录。

---

## 扩展新任务

1. **创建任务目录**
   ```bash
   mkdir -p tasks/<new_task>/cfg/datasets
   mkdir -p tasks/<new_task>/cfg/models
   ```

2. **配置文件**：复制并修改 `train_config.yaml`、数据集和模型配置

3. **脚本文件**：复制并修改 `train_<task>.py`

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [.claude/skills/listenai_quant/README.md](.claude/skills/listenai_quant/README.md) | LISTENAI 工具链使用指南 |
| [thirdparty/README.md](thirdparty/README.md) | 第三方依赖安装说明 |

---

## 支持的硬件平台

- **venusA** (推荐)
- **venus**
- **arcs**