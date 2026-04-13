# Cat Face Detection - 快速上手指南

使用 YOLO26n 在 Oxford-IIIT Pet 数据集上进行猫脸检测训练。

## 环境要求

- Python 3.10+
- PyTorch 2.x (CUDA 12.x)
- NVIDIA GPU (建议 8GB+ VRAM)

## 快速开始

### 1. 激活环境

```bash
# 方式一: 使用已有的 conda 环境
conda activate animal_detection

# 方式二: 如果没有则创建新环境
conda create -n yolo_cat python=3.10 -y
conda activate yolo_cat
pip install torch torchvision
pip install ultralytics
```

### 2. 数据准备

Oxford-IIIT Pet 数据集需提前下载到:
```
/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/
```

运行数据转换脚本:
```bash
cd /CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics
python prepare_cat_dataset.py
```

这将创建:
- `cat_face/train/images/` - 946 张训练图片
- `cat_face/train/labels/` - 训练标注
- `cat_face/val/images/` - 235 张验证图片
- `cat_face/val/labels/` - 验证标注

### 3. 训练模型

```bash
# 确保使用 GPU 2
export CUDA_VISIBLE_DEVICES=2

# 训练 (50 epochs, batch=16)
yolo detect train \
  data=/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/ultralytics/cfg/datasets/cat_face.yaml \
  model=yolo26n.yaml \
  epochs=50 imgsz=640 batch=16 device=2
```

### 4. 验证模型

```bash
# 在验证集上评估
yolo detect val \
  model=/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face/runs/detect/cat_face_yolo26n2/weights/best.pt \
  data=/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics/ultralytics/cfg/datasets/cat_face.yaml
```

### 5. 推理测试

```bash
# 单张图片检测
yolo detect predict \
  model=/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face/runs/detect/cat_face_yolo26n2/weights/best.pt \
  source=/path/to/cat/image.jpg

# 批量检测
yolo detect predict \
  model=/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/cat_face/runs/detect/cat_face_yolo26n2/weights/best.pt \
  source=/path/to/images/
```

### 6. 导出 ONNX 模型

```bash
# 导出为 ONNX 格式 (默认: simplify=True, opset=12, img_size=640)
python export_onnx.py \
  --model cat_face/runs/detect/cat_face_yolo26n2/weights/best.pt \
  --img-size 640 \
  --opset 12

# 不简化 ONNX 模型
python export_onnx.py --model best.pt --no-simplify

# 大尺寸导出
python export_onnx.py --model best.pt --img-size 1280
```

ONNX 模型将保存为: `best.onnx`

### 7. 分析 ONNX 模型

对导出的 ONNX 模型进行全面的内存和性能分析：

```bash
# 基础分析
python analyze_onnx.py yolo26n.onnx

# 完整分析（包含执行时间 profiling）
python analyze_onnx.py yolo26n.onnx --profile --optimized

# 自定义 profiling 参数
python analyze_onnx.py yolo26n.onnx --profile --warmup 5 --runs 20

# 指定输出目录
python analyze_onnx.py yolo26n.onnx --profile --tmp-dir mytmp --output-dir myoutput
```

**分析内容包括：**

1. **权重分析** - 各层参数数量和内存占用
2. **激活分析** - 输入输出张量形状和内存需求
3. **缓存分析** - 需要缓存的张量及总缓存内存
4. **执行内存时间线** - 每个节点执行时的实时内存占用（新增）
   - 已加载的权重内存
   - 当前缓存的激活内存
   - 节点输出内存
   - 总内存占用
5. **执行时间 Profiling** - 各节点执行时间和性能瓶颈
6. **峰值内存估算** - 推理时的峰值内存需求

**输出文件：**
- `output/yolo26n_analysis_YYYYMMDD_HHMMSS.txt` - 完整分析报告
- `tmp/onnx_profile_*.json` - ONNX Runtime profiling 临时数据

**示例输出：**

```
EXECUTION MEMORY TIMELINE
================================================================================
Tracking memory usage at each node execution step...
(Weights accumulate once loaded, activations cached until no longer needed)

Idx   Node Name                                          Weights        Cache       Output        Total
---------------------------------------------------------------------------------------------------------
0     /model.0/conv/Conv                                  1.75KB       6.25MB       6.25MB       6.25MB  <- PEAK
1     /model.0/act/Sigmoid                                1.75KB      12.50MB       6.25MB      12.50MB  <- PEAK
...
180   /model.14/Resize                                    5.65MB       7.42MB       3.12MB      13.07MB  <- PEAK

Peak Memory: 13.07MB at node: /model.14/Resize

Top 10 Memory-Critical Nodes:
Idx   Node Name                                          Weights        Cache       Output        Total
---------------------------------------------------------------------------------------------------------
180   /model.14/Resize                                    5.65MB       7.42MB       3.12MB      13.07MB
...

SUMMARY
================================================================================
Metric                                        Value
----------------------------------------------------
Total Parameters                              2.45M
Weights Memory                               9.35MB
Activation Memory                          269.35MB
Cache Memory                               110.97MB
Number of Nodes                                 384
Number of Cached Tensors                        194
```

## 文件说明

```
ultralytics/
├── prepare_cat_dataset.py              # 数据转换脚本
├── export_onnx.py                      # ONNX 导出脚本
├── analyze_onnx.py                     # ONNX 模型分析工具
├── ultralytics/cfg/datasets/
│   └── cat_face.yaml                   # 数据集配置
├── cat_face_detection_summary.md       # 详细项目总结
├── tmp/                                # 临时文件目录 (profiling 数据等)
├── output/                             # 输出文件目录 (分析报告等)
└── README.cat_detect.md               # 本文档

cat_face/runs/detect/cat_face_yolo26n2/
├── weights/
│   ├── best.pt                         # 最佳权重 (推荐使用)
│   └── last.pt                        # 最后一次权重
├── results.csv                        # 训练指标
└── args.yaml                          # 训练参数
```

## 训练结果

| 指标 | 数值 |
|------|------|
| mAP50 | 0.962 (96.2%) |
| mAP50-95 | 0.795 (79.5%) |
| Precision | 91.5% |
| Recall | 92.8% |

## 自定义训练参数

```bash
# 调整 epochs
epochs=100

# 调整 batch size (根据 GPU 内存)
batch=32  # 需要更多内存

# 调整图像大小
imgsz=1280  # 更大尺寸，更高精度

# 使用更多数据增强
yolo detect train ... \
  augment=True \
  hsv_h=0.015 \
  degrees=10.0 \
  flipud=0.5 \
  fliplr=0.5
```

## 使用训练好的模型 (Python)

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('/path/to/best.pt')

# 预测
results = model('/path/to/cat/image.jpg')

# 显示结果
for r in results:
    print(f"检测到 {len(r.boxes)} 只猫")
    for box in r.boxes:
        print(f"  置信度: {box.conf[0]:.2f}")
```

## 常见问题

### Q: 训练时找不到数据集
A: 确保使用完整绝对路径，不要使用相对路径

### Q: GPU 内存不足
A: 减小 batch size: `batch=8` 或 `batch=4`

### Q: 训练太慢
A: 减少 workers: `workers=4`，或关闭 AMP: `amp=False`

## 注意事项

1. 数据集路径中的所有文件路径都使用绝对路径
2. 训练时会自动下载 YOLO26n 预训练权重 (~5MB)
3. 训练输出默认保存在数据集目录下的 `runs/detect/` 文件夹
4. 推荐使用 `best.pt` 进行推理，它是在验证集上 mAP 最高的模型
5. `tmp/` 目录存放临时文件（profiling 数据等），可定期清理
6. `output/` 目录存放分析报告，建议保留用于性能对比

## 使用本地 ultralytics 仓库（开发调试用）

如果你需要修改 ultralytics 源代码或使用最新功能，可以使用本地仓库：

### 方式一：editable install（推荐）

```bash
# 克隆仓库（如果还没有）
# cd /CodeRepo/Code/dwwang16/workspace/gitee/yolo
# git clone https://github.com/ultralytics/ultralytics.git

# 创建环境并安装
conda create -n yolo_dev python=3.10 -y
conda activate yolo_dev
pip install torch torchvision
cd /CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics
pip install -e .
```

### 方式二：设置 PYTHONPATH

```bash
export PYTHONPATH=/CodeRepo/Code/dwwang16/workspace/gitee/yolo/ultralytics:$PYTHONPATH
```

### 验证本地仓库是否被使用

```python
python -c "import ultralytics; print(ultralytics.__file__)"
# 应该输出包含 ultralytics 目录的路径
```

详细说明见 [README.local_repo_usage.md](./README.local_repo_usage.md)

## 性能优化建议

使用 `analyze_onnx.py` 分析模型后，可以根据结果进行优化：

### 1. 内存优化

**识别内存瓶颈：**
```bash
# 分析内存时间线，找出峰值节点
python analyze_onnx.py model.onnx --profile
```

**优化策略：**
- 峰值出现在早期层：考虑减少输入尺寸或通道数
- Cache 内存过大：检查是否有不必要的 tensor 缓存
- Weights 内存过大：考虑模型剪枝或量化

### 2. 推理速度优化

**分析执行时间：**
```bash
# 分析各节点执行时间
python analyze_onnx.py model.onnx --profile --optimized
```

**优化策略：**
- 识别耗时最长的节点
- 考虑使用 TensorRT 或 OpenVINO 加速
- 调整 batch size 以充分利用 GPU

### 3. 部署优化

**内存估算：**
- 根据分析报告的峰值内存预留足够空间
- 考虑使用 FP16/INT8 量化减少内存占用
- 对于边缘设备，关注 weights + peak activation 的总和

**示例分析：**
```
Peak Memory: 13.07MB (Weights: 5.65MB + Cache: 7.42MB)
部署建议：至少预留 20MB 内存空间（含安全余量）
```
