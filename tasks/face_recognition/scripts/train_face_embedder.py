"""
Face Embedder Training Script
Lightweight face feature extractor training with ArcFace loss
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

class FaceRecognitionDataset(Dataset):
    """Dataset for face recognition training"""

    def __init__(self, data_dir, img_size=112, augment=True, num_identities=500):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment

        # Load identity folders
        self.samples = []
        self.labels = []
        self._load_samples(num_identities)

        # Get unique identities
        self.unique_labels = sorted(list(set(self.labels)))
        self.num_classes = len(self.unique_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

        print(f"Loaded {len(self.samples)} images from {self.num_classes} identities")

    def _load_samples(self, max_identities=None):
        """Load image paths and identity labels"""
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return

        identity_dirs = [d for d in os.listdir(self.data_dir)
                        if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('id_')]

        if max_identities:
            identity_dirs = identity_dirs[:max_identities]

        for identity_dir in identity_dirs:
            identity_path = os.path.join(self.data_dir, identity_dir)
            identity_id = identity_dir  # e.g., "id_0001"

            for img_file in os.listdir(identity_path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    img_path = os.path.join(identity_path, img_file)
                    self.samples.append(img_path)
                    self.labels.append(identity_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            # Return a blank face if loading fails
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Data augmentation
        if self.augment:
            img = self._augment(img)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]

        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # Convert label to index
        label_idx = self.label_to_idx[label]

        return img, label_idx

    def _augment(self, img):
        """Data augmentation for face images"""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()

        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)

        # Random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 1)

        # Random small rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            center = (self.img_size // 2, self.img_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (self.img_size, self.img_size),
                                borderMode=cv2.BORDER_REFLECT)

        return (img * 255).astype(np.uint8)


def train_embedder(args):
    """Train face embedder"""
    print("=" * 60)
    print("Face Embedder Training")
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
    from face_models import FaceEmbedder

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = FaceEmbedder(
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        input_size=args.input_size,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Create dataset
    train_dataset = FaceRecognitionDataset(
        data_dir=args.train_dir,
        img_size=args.input_size,
        augment=True,
        num_identities=args.num_classes
    )

    # Update num_classes based on actual data
    actual_num_classes = train_dataset.num_classes
    if actual_num_classes < args.num_classes:
        print(f"Updating num_classes from {args.num_classes} to {actual_num_classes}")
        args.num_classes = actual_num_classes

        # Rebuild model with correct num_classes
        model = FaceEmbedder(
            embedding_dim=args.embedding_dim,
            num_classes=args.num_classes,
            input_size=args.input_size,
            dropout=args.dropout
        ).to(device)

    # Handle empty dataset
    if len(train_dataset) == 0:
        print("No training data found! Creating synthetic data...")
        create_synthetic_training_data(args.train_dir, args.num_classes)

        # Reload dataset
        train_dataset = FaceRecognitionDataset(
            data_dir=args.train_dir,
            img_size=args.input_size,
            augment=True,
            num_identities=args.num_classes
        )

    # Create a simple sampler that repeats samples if needed
    if len(train_dataset) < args.batch_size:
        print(f"Dataset size ({len(train_dataset)}) is smaller than batch size ({args.batch_size})")
        print("Using repeated sampling for training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
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

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            logits = model(images, labels)

            # Loss
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        scheduler.step()

        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Acc = {train_acc:.2f}%")

        # Save best model
        if train_acc > best_acc:
            best_acc = train_acc
            save_path = os.path.join(args.output, "best_embedder.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': train_acc,
                'num_classes': args.num_classes,
                'embedding_dim': args.embedding_dim,
            }, save_path)
            print(f"  Saved best model to {save_path}")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.output, f"embedder_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, save_path)

    # Save label mapping
    label_mapping = {
        'label_to_idx': train_dataset.label_to_idx,
        'idx_to_label': {v: k for k, v in train_dataset.label_to_idx.items()}
    }
    with open(os.path.join(args.output, "label_mapping.json"), 'w') as f:
        json.dump(label_mapping, f, indent=2)

    print("\nTraining complete!")
    print(f"Best model saved to: {os.path.join(args.output, 'best_embedder.pth')}")
    print(f"Best accuracy: {best_acc:.2f}%")


def create_synthetic_training_data(output_dir, num_identities):
    """Create synthetic face recognition training data"""
    print(f"Creating synthetic training data at {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Create identity folders with sample images
    for identity in range(num_identities):
        identity_dir = os.path.join(output_dir, f"id_{identity:04d}")
        os.makedirs(identity_dir, exist_ok=True)

        # Create 10-15 sample images per identity
        num_images = random.randint(10, 15)
        for img_idx in range(num_images):
            # Generate a simple face-like pattern
            img = np.zeros((112, 112, 3), dtype=np.uint8)

            # Random skin tone
            skin_tone = (
                random.randint(180, 220),
                random.randint(150, 190),
                random.randint(130, 170)
            )
            img[:, :] = skin_tone

            # Add identity-specific features (random noise seed)
            np.random.seed(identity * 1000 + img_idx)
            noise = np.random.normal(0, 10, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

            # Save image
            img_path = os.path.join(identity_dir, f"img_{img_idx:03d}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Created {num_identities} identities with synthetic face images")


def main():
    parser = argparse.ArgumentParser(description="Train Face Embedder")
    parser.add_argument("--train-dir", type=str, required=True,
                       help="Training data directory")
    parser.add_argument("--output", type=str, default="runs/face_recognition/embedder",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=80,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--input-size", type=int, default=112,
                       help="Input image size")
    parser.add_argument("--embedding-dim", type=int, default=128,
                       help="Embedding dimension")
    parser.add_argument("--num-classes", type=int, default=500,
                       help="Number of identities")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
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
    train_embedder(args)


if __name__ == "__main__":
    main()