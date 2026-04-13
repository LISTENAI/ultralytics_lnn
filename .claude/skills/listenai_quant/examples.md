# LISTENAI 量化训练代码示例

本文档提供 linger 量化 API 的使用示例。

> **注意**：配置文件路径使用 `<task_dir>/cfg/` 格式

---

## 1. 浮点约束训练

```python
import linger

# 加载模型
model = YourModel()

# 添加浮点约束
config_file = 'tasks/<task_dir>/cfg/quant_config.yaml'
model = linger.constrain(model, config_file=config_file)

# 正常训练...
```

---

## 2. 量化微调

```python
import linger

# 加载模型
model = YourModel()

# 添加量化
config_file = 'tasks/<task_dir>/cfg/quant_config.yaml'
model = linger.init(model, config_file=config_file)

# 继续训练...
```

---

## 3. 导出量化 ONNX

```python
import torch
import linger

model = YourModel()
model = linger.init(model, config_file='tasks/<task_dir>/cfg/quant_config.yaml')
model.eval()

dummy_input = torch.randn(1, 3, 320, 320)

with torch.no_grad():
    linger.onnx.export(
        model, dummy_input, "model_quant.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"]
    )
```

---

## 4. YOLO 量化训练

```python
from ultralytics import YOLO
import linger

model = YOLO('tasks/<task_dir>/cfg/models/yolo26n_cat.yaml')

# 添加量化，跳过不支持的模块
model.model = linger.init(
    model.model,
    config_file='tasks/<task_dir>/cfg/quant_config.yaml',
    disable_submodel=('*9.m',)  # 跳过 SPPF
)

# 训练
results = model.train(
    data='tasks/<task_dir>/cfg/datasets/cat_face_320.yaml',
    epochs=20,
    imgsz=320
)
```

---

## 5. 配置文件说明

`tasks/<task_dir>/cfg/train_config.yaml`:

```yaml
epochs: 50
imgsz: 320
batch: 32
device: 2
lr: 0.01
quant: false
```

---

## 6. 常见问题

### MaxPool 不支持
```python
model = linger.init(model, disable_submodel=('*9.m',))
```