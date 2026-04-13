#!/usr/bin/env python3
"""
生成测试图像用于推理测试
"""

import cv2
import numpy as np
from pathlib import Path

# 尝试导入qrcode库
try:
    import qrcode

    def generate_qr(content, size=200):
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,
                          box_size=10, border=2)
        qr.add_data(content)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = np.array(img)
        if size != img.shape[0]:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        return img
except ImportError:
    def generate_qr(content, size=200):
        # 简单的模拟二维码
        img = np.ones((size, size), dtype=np.uint8) * 255
        box = size // 7
        # 三个定位图案
        for x, y in [(box, box), (size-box*3, box), (box, size-box*3)]:
            cv2.rectangle(img, (x, y), (x+box*3, y+box*3), 0, -1)
            cv2.rectangle(img, (x+box, y+box), (x+box*2, y+box*2), 255, -1)
        return img

# 创建测试图像
output_dir = Path(__file__).parent.parent / 'test_images'
output_dir.mkdir(exist_ok=True)

contents = [
    "https://example.com/test1",
    "HELLO_WORLD",
    "1234567890",
    "QR_TEST_DATA",
]

for i, content in enumerate(contents):
    # 生成二维码
    qr = generate_qr(content, 150)

    # 创建背景图像
    img = np.ones((320, 320, 3), dtype=np.uint8) * 240

    # 放置二维码 (转为RGB)
    x, y = 85, 85
    for c in range(3):
        img[y:y+150, x:x+150, c] = qr

    # 添加文字
    cv2.putText(img, f"Test QR {i+1}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (50, 50, 50), 2)

    # 保存
    path = output_dir / f"test_qr_{i+1}.jpg"
    cv2.imwrite(str(path), img)
    print(f"Generated: {path}")

print(f"\n生成 {len(contents)} 张测试图像")