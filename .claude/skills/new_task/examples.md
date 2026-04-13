# 任务创建示例

## 示例 1: 创建猫脸检测任务

### 步骤 1: 复制模板

```bash
# 复制模板目录
cp -r tasks/cat_320X320_detect tasks/cat_face_detection

# 重命名文件
mv tasks/cat_face_detection/train_cat_face_320.py tasks/cat_face_detection/train_cat.py
mv tasks/cat_face_detection/test_cat_face_320.py tasks/cat_face_detection/test_cat.py
```

### 步骤 2: 修改配置文件

修改 `tasks/cat_face_detection/cfg/train_config.yaml`:

```yaml
epochs: 50
imgsz: 320
batch: 32
device: 0
project: ./runs/detect
name: cat_face_detect
dataset: cat_face_320.yaml
model: yolo26n_cat.yaml
```

### 步骤 3: 准备数据集

确保数据集结构如下:

```
dataset/
├── train/
│   └── images/
│       ├── cat_001.jpg
│       └── cat_001.txt  # YOLO格式标注
├── val/
│   └── images/
└── test/
    └── images/
```

### 步骤 4: 运行训练

```bash
python tasks/cat_face_detection/train_cat.py
```

---

## 示例 2: 创建车辆检测任务

### 步骤 1: 创建目录结构

```bash
mkdir -p tasks/vehicle_detection/cfg/datasets
mkdir -p tasks/vehicle_detection/cfg/models
```

### 步骤 2: 配置文件

**train_config.yaml**:
```yaml
epochs: 100
imgsz: 640
batch: 16
device: 0
lr: 0.01
project: ./runs/detect
name: vehicle_yolo26n
dataset: vehicle.yaml
model: yolo26n_vehicle.yaml
```

**datasets/vehicle.yaml**:
```yaml
path: /path/to/vehicle_dataset
train: train/images
val: val/images

nc: 3
names:
  0: car
  1: truck
  2: bus
```

**models/yolo26n_vehicle.yaml**:
```yaml
# 基础模型配置
nc: 3
names:
  0: car
  1: truck
  2: bus
```

### 步骤 3: 创建训练脚本

复制并修改训练脚本，主要修改：

```python
# 1. 配置路径
TASK_NAME = "vehicle_detection"
CONFIG_FILE = CFG_DIR / "train_config.yaml"

# 2. 数据集和模型配置路径
data_config = CFG_DIR / "datasets" / "vehicle.yaml"
model_config = CFG_DIR / "models" / "yolo26n_vehicle.yaml"
```

### 步骤 4: 运行

```bash
python tasks/vehicle_detection/train_vehicle.py --epochs 10
```

---

## 示例 3: 创建图像分割任务

### 步骤 1: 修改配置文件

```yaml
# train_config.yaml
task: segment
imgsz: 320
model: yolo26n-seg.yaml
```

### 步骤 2: 数据集配置添加分割标注

```yaml
# datasets/segment.yaml
path: /path/to/seg_dataset
train: train/images
val: val/images

nc: 2
names:
  0: cat
  1: dog
```

### 步骤 3: 标注格式

使用 YOLO 分割标注格式 (多边形):

```
# cat_001.txt (每个目标一行)
0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2  # 类别ID + 归一化坐标点
```

### 步骤 4: 运行分割训练

```bash
python tasks/<task>/train_<task>.py --task segment
```

---

## 示例 4: 创建量化训练任务

### 步骤 1: 浮点训练

```bash
python tasks/<task>/train_<task>.py --epochs 50
```

### 步骤 2: 量化约束训练

```bash
python tasks/<task>/train_<task>.py --quant --quant-stage constrain --epochs 10
```

### 步骤 3: 量化训练

```bash
python tasks/<task>/train_<task>.py --quant --quant-stage quant --epochs 5
```

### 步骤 4: 导出并打包

```bash
# 导出 ONNX
python tasks/<task>/test_<task>.py --export-onnx

# 优化 ONNX
python tasks/pack_model.py --model best.onnx --optimize-onnx

# 打包模型
python tasks/pack_model.py --model best_int8.onnx --platform venusA
```

---

## 常见问题

### Q: 如何选择模型大小?

| 模型 | 参数量 | 适用场景 |
|------|--------|----------|
| yolo26n | 2.6M | 资源受限环境 |
| yolo26s | 7.2M | 平衡性能与速度 |
| yolo26m | 25.9M | 高精度需求 |

### Q: 如何确定输入图像尺寸?

- 人脸/小目标检测: 320x320 或 416x416
- 通用目标检测: 640x640
- 高精度需求: 1280x1280

### Q: 如何调整学习率?

```yaml
lr: 0.01        # 初始学习率
lrf: 0.01       # 最终学习率 (lr * lrf)
```

### Q: 训练时内存不足怎么办?

```yaml
batch: 8        # 减小批次大小
imgsz: 320      # 减小图像尺寸
amp: false      # 禁用混合精度
```