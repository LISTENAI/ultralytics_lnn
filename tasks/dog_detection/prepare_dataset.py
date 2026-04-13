#!/usr/bin/env python3
"""
Dog Face Dataset Converter - 使用已有的 Oxford-IIIT Pet Dataset
"""

import os
import shutil
import random
from pathlib import Path
import xml.etree.ElementTree as ET


# 狗品种列表
DOG_BREEDS = {
    'staffordshire_bull_terrier', 'scottish_terrier', 'yorkshire_terrier',
    'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter',
    'german_shorthaired', 'great_pyrenees', 'beagle', 'saint_bernard',
    'shiba_inu', 'samoyed', 'japanese_chin', 'maltese_dog', 'pug'
}

# 数据集根目录
DATASET_ROOT = Path("/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset")
OUTPUT_DIR = Path("/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset/dog_face")

IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations" / "xmls"


def convert_bbox_to_yolo(xml_path: Path, img_width: int, img_height: int):
    """将 VOC bbox 转换为 YOLO 格式"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()

        # Oxford-IIIT Pet 数据集中，name 就是 "dog"
        if name != 'dog':
            continue

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 狗作为类别 0
        annotations.append((0, x_center, y_center, width, height))

    return annotations


def main():
    # 创建输出目录
    train_img_dir = OUTPUT_DIR / "train" / "images"
    train_label_dir = OUTPUT_DIR / "train" / "labels"
    val_img_dir = OUTPUT_DIR / "val" / "images"
    val_label_dir = OUTPUT_DIR / "val" / "labels"

    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 收集所有狗的图片
    dog_files = []
    for xml_file in ANNOTATIONS_DIR.glob("*.xml"):
        breed = xml_file.stem.rsplit('_', 1)[0].lower()
        if breed in DOG_BREEDS:
            dog_files.append(xml_file)

    print(f"Found {len(dog_files)} dog images")

    # 分割训练集和验证集 (80/20)
    random.seed(42)
    random.shuffle(dog_files)
    val_size = int(len(dog_files) * 0.2)
    val_files = dog_files[:val_size]
    train_files = dog_files[val_size:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # 处理训练集
    for i, xml_file in enumerate(train_files):
        img_name = xml_file.stem + ".jpg"
        img_file = IMAGES_DIR / img_name

        if not img_file.exists():
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        annotations = convert_bbox_to_yolo(xml_file, img_width, img_height)
        if not annotations:
            continue

        shutil.copy2(img_file, train_img_dir / img_name)

        label_file = train_label_dir / (xml_file.stem + ".txt")
        with open(label_file, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

    # 处理验证集
    for xml_file in val_files:
        img_name = xml_file.stem + ".jpg"
        img_file = IMAGES_DIR / img_name

        if not img_file.exists():
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        annotations = convert_bbox_to_yolo(xml_file, img_width, img_height)
        if not annotations:
            continue

        shutil.copy2(img_file, val_img_dir / img_name)

        label_file = val_label_dir / (xml_file.stem + ".txt")
        with open(label_file, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

    # 创建数据集配置文件
    dataset_yaml = OUTPUT_DIR / "dog.yaml"
    dataset_yaml.write_text(f"""# Dog Face Detection Dataset
path: {OUTPUT_DIR}
train: train/images
val: val/images

nc: 1
names:
  0: dog
""")

    print(f"\nDataset ready: {OUTPUT_DIR}")
    print(f"Train: {len(list(train_img_dir.glob('*.jpg')))} images")
    print(f"Val: {len(list(val_img_dir.glob('*.jpg')))} images")
    print(f"Config: {dataset_yaml}")


if __name__ == "__main__":
    main()