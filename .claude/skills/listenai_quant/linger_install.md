# linger 与 torch 版本兼容性说明

## 当前环境

- **torch**: 2.10.0
- **linger**: 3.0.6
- **状态**: ❌ 不兼容

## 问题描述

导入 linger 时出现以下错误：

```
ImportError: cannot import name 'prim_ConstantChunk' from 'torch.onnx.symbolic_opset9'
```

这是因为 PyTorch 2.10.0 移除了 `prim_ConstantChunk` 符号，而 linger 3.0.6 仍然尝试导入它。

## 解决方案

### 方案1: 降级 PyTorch（推荐）

```bash
# 卸载当前版本
pip uninstall torch torchvision

# 安装兼容版本 (PyTorch 2.0.x - 2.3.x)
pip install torch==2.3.0 torchvision==0.18.0
```

### 方案2: 升级 linger（如果可用）

检查 linger 是否有新版本修复了这个问题：

```bash
pip install linger --upgrade
```

### 方案3: 使用 Docker 环境

linger 官方提供了 Docker 环境，已配置好兼容的依赖：

```bash
# 参考 thirdparty/linger/doc/tutorial/install.md 中的 Docker 安装方式
```

## 验证安装

安装正确后，运行以下命令验证：

```python
import linger
from linger import init, constrain
print("linger 导入成功!")
```

## 量化训练流程

安装成功后，可以使用以下命令进行训练：

```bash
# 1. 浮点训练
python tasks/qrcode_detect/scripts/train_qr_detector.py --epochs 50

# 2. 约束训练
python tasks/qrcode_detect/scripts/train_qr_detector.py --quant --quant-stage constrain

# 3. 量化训练
python tasks/qrcode_detect/scripts/train_qr_detector.py --quant --quant-stage quant

# 4. 一键量化训练（约束+量化）
python tasks/qrcode_detect/scripts/train_qr_detector.py --quant --quant-stage both
```