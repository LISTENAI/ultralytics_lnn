#!/usr/bin/env python3
"""
快速生成一个用于测试的模型权重文件
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qrcode_models import QRDetector

# 创建模型
model = QRDetector(in_channels=3, num_angle_classes=4)

# 创建dummy checkpoint
checkpoint = {
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': {},
    'loss': 0.5,
}

# 保存
output_path = Path(__file__).parent / 'runs' / 'qrcode_detect' / 'best_detector.pth'
output_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(checkpoint, output_path)
print(f"Model saved to {output_path}")

# 验证
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
print(f"Parameter constraint: < 4M ✓" if params < 4000000 else "FAIL")