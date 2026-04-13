#!/usr/bin/env python3
"""
Convert Oxford-IIIT Pet Dataset cat face annotations to YOLO format.
"""

import os
import shutil
import random
from pathlib import Path

# Paths
DATASET_ROOT = Path("/CodeRepo/Code/dwwang16/dataset/animal_detection/Oxford-IIIT_Pet_Dataset")
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
IMAGES_DIR = DATASET_ROOT / "images"
OUTPUT_DIR = DATASET_ROOT / "cat_face"

TRAINVAL_FILE = ANNOTATIONS_DIR / "trainval.txt"
TRAIN_OUTPUT = OUTPUT_DIR / "train"
VAL_OUTPUT = OUTPUT_DIR / "val"

# Filter cats (SPECIES=1)
def load_cat_samples():
    """Load cat samples from trainval.txt (SPECIES=1 = cat)."""
    cats = []
    with open(TRAINVAL_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                filename = parts[0]
                species = int(parts[2])  # 1=cat, 2=dog
                if species == 1:  # Only cats
                    cats.append(filename)
    return cats

def convert_voc_to_yolo(xml_path, output_path):
    """Convert Pascal VOC XML to YOLO format."""
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except:
        return False

    # Get image size
    size = root.find('size')
    if size is None:
        return False
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # Find cat objects (head bounding box)
    yolo_lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name != 'cat':
            continue

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Convert to YOLO format (normalized)
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        # Class 0 = cat
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if yolo_lines:
        with open(output_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        return True
    return False

def main():
    print("Loading cat samples from trainval.txt...")
    cat_samples = load_cat_samples()
    print(f"Found {len(cat_samples)} cat samples")

    # Shuffle for random train/val split
    random.seed(42)
    random.shuffle(cat_samples)

    # Split: 80% train, 20% val
    split_idx = int(len(cat_samples) * 0.8)
    train_samples = cat_samples[:split_idx]
    val_samples = cat_samples[split_idx:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Process train set
    print("\nProcessing training set...")
    train_count = 0
    for i, sample in enumerate(train_samples):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(train_samples)}")

        img_file = IMAGES_DIR / f"{sample}.jpg"
        xml_file = ANNOTATIONS_DIR / "xmls" / f"{sample}.xml"

        if not img_file.exists() or not xml_file.exists():
            continue

        # Convert annotation
        label_file = TRAIN_OUTPUT / "labels" / f"{sample}.txt"
        if convert_voc_to_yolo(xml_file, label_file):
            # Copy image
            dest_img = TRAIN_OUTPUT / "images" / f"{sample}.jpg"
            shutil.copy(img_file, dest_img)
            train_count += 1

    print(f"Processed {train_count} training samples")

    # Process val set
    print("\nProcessing validation set...")
    val_count = 0
    for i, sample in enumerate(val_samples):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(val_samples)}")

        img_file = IMAGES_DIR / f"{sample}.jpg"
        xml_file = ANNOTATIONS_DIR / "xmls" / f"{sample}.xml"

        if not img_file.exists() or not xml_file.exists():
            continue

        # Convert annotation
        label_file = VAL_OUTPUT / "labels" / f"{sample}.txt"
        if convert_voc_to_yolo(xml_file, label_file):
            # Copy image
            dest_img = VAL_OUTPUT / "images" / f"{sample}.jpg"
            shutil.copy(img_file, dest_img)
            val_count += 1

    print(f"Processed {val_count} validation samples")
    print(f"\nDone! Total: train={train_count}, val={val_count}")

if __name__ == "__main__":
    main()
