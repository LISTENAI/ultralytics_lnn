"""
Face Detector Training Script
Lightweight face detector training with anchor-free detection
"""

import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import random
import json
from tqdm import tqdm

# ==================== linger 量化训练支持 ====================
# 添加 linger 路径
SCRIPT_DIR = Path(__file__).parent.resolve()
TASK_DIR = SCRIPT_DIR.parent
ULTRALYTICS_PATH = TASK_DIR.parent.parent
LINGER_PATH = ULTRALYTICS_PATH / 'thirdparty' / 'linger'
if LINGER_PATH.exists() and str(LINGER_PATH) not in sys.path:
    sys.path.insert(0, str(LINGER_PATH))

# 清理 linger 模块缓存
if 'linger' in sys.modules:
    del sys.modules['linger']
for mod in list(sys.modules.keys()):
    if mod.startswith('linger'):
        del sys.modules[mod]

# linger 可用性检查
LINGER_AVAILABLE = False
try:
    import linger
    from linger import init, constrain
    LINGER_AVAILABLE = True
except ImportError:
    pass


def add_quantization(model, config_file=None, stage='quant'):
    """在模型中添加量化"""
    if not LINGER_AVAILABLE:
        print("⚠️ linger 不可用，跳过量量化")
        return model

    if stage == 'constrain':
        print(f"🔧 添加浮点约束训练...")
        return constrain(model, config_file=config_file)
    else:
        print(f"🔧 添加量化训练...")
        return init(model, config_file=config_file)


class FaceDetectionDataset(Dataset):
    """Dataset for face detection training"""

    def __init__(self, data_dir, img_size=224, augment=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment

        # Load image paths and annotations
        self.samples = []
        self._load_samples()

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def _load_samples(self):
        """Load image paths and bounding box annotations"""
        img_dir = os.path.join(self.data_dir, "images")
        if not os.path.exists(img_dir):
            # Try alternative paths
            for root, dirs, files in os.walk(self.data_dir):
                for f in files:
                    if f.endswith('.jpg'):
                        img_path = os.path.join(root, f)
                        anno_path = img_path.replace('.jpg', '.txt').replace('/images', '')
                        if os.path.exists(anno_path):
                            self.samples.append((img_path, anno_path))
        else:
            # Standard structure
            for f in os.listdir(img_dir):
                if f.endswith('.jpg'):
                    img_path = os.path.join(img_dir, f)
                    # Check multiple possible annotation paths
                    anno_candidates = [
                        img_path.replace('.jpg', '.txt'),
                        img_path.replace('/images/', '/labels/').replace('.jpg', '.txt'),
                        img_path.replace('images', 'labels').replace('.jpg', '.txt'),
                    ]
                    anno_path = None
                    for candidate in anno_candidates:
                        if os.path.exists(candidate):
                            anno_path = candidate
                            break
                    if anno_path:
                        self.samples.append((img_path, anno_path))

        # If no samples found, try to generate from synthetic data structure
        if len(self.samples) == 0:
            for root, dirs, files in os.walk(self.data_dir):
                for f in files:
                    if f.endswith('.jpg'):
                        img_path = os.path.join(root, f)
                        # Try to find annotation in parent directories
                        rel_path = os.path.relpath(img_path, self.data_dir)
                        parts = rel_path.split(os.sep)
                        if len(parts) > 1:
                            anno_path = os.path.join(self.data_dir, parts[0], parts[1].replace('.jpg', '.txt'))
                            if os.path.exists(anno_path):
                                self.samples.append((img_path, anno_path))

    def _parse_annotation(self, anno_path):
        """Parse annotation file"""
        bboxes = []
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # class_id x y w h - support both pixel and YOLO format
                        try:
                            x, y, w, h = map(float, parts[1:5])
                            bboxes.append([x, y, w, h])
                        except ValueError:
                            # Try as integer (pixel format)
                            x, y, w, h = map(int, parts[1:5])
                            bboxes.append([x, y, w, h])
        return bboxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, anno_path = self.samples[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            # Return a blank image if loading fails
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        # Load bounding boxes
        bboxes = self._parse_annotation(anno_path)

        # Data augmentation
        if self.augment:
            img, bboxes = self._augment(img, bboxes)

        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]

        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # Generate ground truth
        cls_targets, box_targets = self._generate_targets(bboxes, h, w)

        return img, cls_targets, box_targets

    def _augment(self, img, bboxes):
        """Data augmentation"""
        # Horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            for i, box in enumerate(bboxes):
                bboxes[i][0] = img.shape[1] - box[0] - box[2]

        # Brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Contrast
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

        return img, bboxes

    def _generate_targets(self, bboxes, img_h, img_w):
        """Generate ground truth for multi-scale detection"""
        # Generate targets for 3 scales
        cls_targets = []
        box_targets = []

        # Scales based on input size (FPN produces 1/4, 1/8, 1/16 of input)
        input_size = self.img_size
        scales = [input_size // 4, input_size // 8, input_size // 16]  # [80, 40, 20] for 320 input

        for scale in scales:
            # Create ground truth maps
            cls_map = torch.zeros(1, scale, scale)
            box_map = torch.zeros(4, scale, scale)

            # Map bboxes to this scale
            scale_factor = scale / max(img_h, img_w)

            for box in bboxes:
                x, y, w, h = box
                # Convert to center format
                cx = (x + w / 2) * scale_factor
                cy = (y + h / 2) * scale_factor
                bw = w * scale_factor
                bh = h * scale_factor

                # Map to grid
                cx_int = int(cx)
                cy_int = int(cy)

                if 0 <= cx_int < scale and 0 <= cy_int < scale:
                    # Classification target (positive)
                    cls_map[0, cy_int, cx_int] = 1.0

                    # Regression target (normalized)
                    box_map[0, cy_int, cx_int] = (cx - cx_int) / 1.0  # dx
                    box_map[1, cy_int, cx_int] = (cy - cy_int) / 1.0  # dy
                    box_map[2, cy_int, cx_int] = np.log(bw + 1e-6)     # log(dw)
                    box_map[3, cy_int, cx_int] = np.log(bh + 1e-6)    # log(dh)

            cls_targets.append(cls_map)
            box_targets.append(box_map)

        return cls_targets, box_targets


class FocalLoss(nn.Module):
    """Focal Loss for face detection"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Focal weight
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gce

        # Apply alpha
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        return (fce * focal_weight).mean()


class DetectionLoss(nn.Module):
    """Combined detection loss"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.l1 = nn.SmoothL1Loss(reduction='mean')

    def forward(self, cls_preds, box_preds, cls_targets, box_targets):
        cls_loss = 0
        box_loss = 0

        for cls_pred, box_pred, cls_target, box_target in zip(cls_preds, box_preds, cls_targets, box_targets):
            # Classification loss (positive samples only)
            pos_mask = cls_target > 0  # [B, 1, H, W]
            if pos_mask.sum() > 0:
                cls_loss += self.bce(cls_pred[pos_mask], cls_target[pos_mask])

                # Box regression loss (positive samples only)
                # Expand mask to match box_pred channels [B, 4, H, W]
                pos_mask_expanded = pos_mask.expand_as(box_pred)
                box_loss += self.l1(box_pred[pos_mask_expanded], box_target[pos_mask_expanded])

        return cls_loss + 0.5 * box_loss


def train_detector(args):
    """Train face detector"""
    print("=" * 60)
    print("Face Detector Training")
    print("=" * 60)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Build model
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_models import FaceDetector

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    model = FaceDetector(num_classes=1, input_size=args.input_size).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create dataset
    train_dataset = FaceDetectionDataset(
        data_dir=args.train_dir,
        img_size=args.input_size,
        augment=True
    )

    val_dataset = FaceDetectionDataset(
        data_dir=args.val_dir,
        img_size=args.input_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Loss
    criterion = DetectionLoss()

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (images, cls_targets, box_targets) in enumerate(pbar):
            images = images.to(device)
            cls_targets = [t.to(device) for t in cls_targets]
            box_targets = [t.to(device) for t in box_targets]

            # Forward
            cls_preds, box_preds = model(images)

            # Loss
            loss = criterion(cls_preds, box_preds, cls_targets, box_targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        if (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, cls_targets, box_targets in val_loader:
                    images = images.to(device)
                    cls_targets = [t.to(device) for t in cls_targets]
                    box_targets = [t.to(device) for t in box_targets]

                    cls_preds, box_preds = model(images)
                    loss = criterion(cls_preds, box_preds, cls_targets, box_targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(args.output, "best_detector.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, save_path)
                print(f"  Saved best model to {save_path}")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.output, f"detector_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, save_path)

    print("\nTraining complete!")
    print(f"Best model saved to: {os.path.join(args.output, 'best_detector.pth')}")


def main():
    parser = argparse.ArgumentParser(description="Train Face Detector")
    parser.add_argument("--train-dir", type=str, required=True,
                       help="Training data directory")
    parser.add_argument("--val-dir", type=str, default=None,
                       help="Validation data directory")
    parser.add_argument("--output", type=str, default="runs/face_recognition/detector",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=80,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--input-size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--val-freq", type=int, default=5,
                       help="Validation frequency")
    parser.add_argument("--save-freq", type=int, default=10,
                       help="Model save frequency")
    parser.add_argument("--use-cuda", action="store_true", default=True,
                       help="Use CUDA if available")

    # 量化训练参数
    parser.add_argument("--quant", action="store_true", help='Enable quantization training')
    parser.add_argument("--quant-stage", type=str, default='both',
                       choices=['both', 'constrain', 'quant'],
                       help='Quantization stage: both(constrain+quant), constrain only, quant only')
    parser.add_argument("--quant-config", type=str, default=None,
                       help='Quantization config file path')

    args = parser.parse_args()

    # Use same directory for validation if not specified
    if args.val_dir is None:
        args.val_dir = args.train_dir

    train_detector(args)


if __name__ == "__main__":
    main()