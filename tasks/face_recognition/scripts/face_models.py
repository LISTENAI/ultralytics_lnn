"""
Face Recognition Models - Lightweight Face Detection and Embedding
Designed for embedded devices with <4M total parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class ConvBNReLU(nn.Module):
    """Depthwise separable convolution with batch normalization and ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (MobileNetV2 style)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, expand_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, padding=0))

        # Depthwise
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim))

        # Linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Lite(nn.Module):
    """Lightweight MobileNetV2 backbone for face models"""

    def __init__(self, in_channels=3, channels=[16, 24, 32, 96, 320], expand_ratio=4, dropout=0.2):
        super().__init__()
        self.channels = channels

        # Initial convolution
        self.stem = ConvBNReLU(in_channels, channels[0], kernel_size=3, stride=2, padding=1)

        # Build stages
        self.stage1 = self._make_stage(channels[0], channels[1], 1, expand_ratio)
        self.stage2 = self._make_stage(channels[1], channels[2], 2, expand_ratio)
        self.stage3 = self._make_stage(channels[2], channels[3], 2, expand_ratio)
        self.stage4 = self._make_stage(channels[3], channels[4], 2, expand_ratio)

        self.out_channels = channels

    def _make_stage(self, in_channels, out_channels, stride, expand_ratio):
        return DepthwiseSeparableConv(in_channels, out_channels, stride=stride, expand_ratio=expand_ratio)

    def forward(self, x):
        """
        Returns multi-scale features for FPN
        x: [B, 3, H, W]
        """
        c1 = self.stage1(self.stem(x))  # [B, 24, H/4, W/4]
        c2 = self.stage2(c1)             # [B, 32, H/8, W/8]
        c3 = self.stage3(c2)             # [B, 96, H/16, W/16]
        c4 = self.stage4(c3)             # [B, 320, H/32, W/32]

        return [c2, c3, c4]


class FPNNeck(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""

    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            lateral_conv = nn.Conv2d(in_ch, out_channels, 1)
            fpn_conv = DepthwiseSeparableConv(out_channels, out_channels, expand_ratio=4)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, features):
        """
        features: List of feature maps from backbone [c2, c3, c4]
        """
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')

        # Output features
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return outs


class DetectionHead(nn.Module):
    """Anchor-free detection head for face detection"""

    def __init__(self, num_classes=1, in_channels=64, feat_channels=64):
        super().__init__()
        self.num_classes = num_classes

        # Classification branch
        self.cls_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, feat_channels, expand_ratio=4),
            DepthwiseSeparableConv(feat_channels, feat_channels, expand_ratio=4),
            nn.Conv2d(feat_channels, num_classes, 1)
        )

        # Regression branch (bbox delta)
        self.reg_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, feat_channels, expand_ratio=4),
            DepthwiseSeparableConv(feat_channels, feat_channels, expand_ratio=4),
            nn.Conv2d(feat_channels, 4, 1)  # dx, dy, dw, dh
        )

    def forward(self, x):
        cls_score = self.cls_conv(x)
        bbox_pred = self.reg_conv(x)
        return cls_score, bbox_pred


class FaceDetector(nn.Module):
    """Lightweight Face Detector (~1.4M parameters)"""

    def __init__(self, num_classes=1, input_size=224):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # Backbone
        self.backbone = MobileNetV2Lite(in_channels=3, channels=[16, 24, 32, 96, 320], expand_ratio=4)

        # FPN Neck
        self.neck = FPNNeck(in_channels_list=[32, 96, 320], out_channels=64)

        # Detection heads for multi-scale
        self.detect_heads = nn.ModuleList([
            DetectionHead(num_classes, 64, 64) for _ in range(3)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Multi-scale feature extraction
        backbone_features = self.backbone(x)  # [c2, c3, c4]
        fpn_features = self.neck(backbone_features)

        # Multi-scale detection
        cls_scores = []
        bbox_preds = []

        for i, feat in enumerate(fpn_features):
            cls_score, bbox_pred = self.detect_heads[i](feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ArcFaceHead(nn.Module):
    """ArcFace (Additive Angular Margin Loss) head for face recognition"""

    def __init__(self, embedding_dim=128, num_classes=10000, margin=0.5, scale=30.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Easy margin parameters
        self.easy_margin = True

    def forward(self, embeddings, labels=None):
        """
        embeddings: [B, embedding_dim]
        labels: [B] (optional, for training)
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)

        if labels is None:
            # Inference mode
            return cosine

        # Training mode with ArcFace loss
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        if self.easy_margin:
            # Easy margin: only increase the margin for correctly classified samples
            target_logits = torch.cos(torch.clamp(theta + self.margin, max=1.57079632679))
        else:
            target_logits = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, self.num_classes).float()
        logits = torch.where(one_hot == 1, target_logits, cosine)
        logits = logits * self.scale

        return logits


class FaceEmbedder(nn.Module):
    """Lightweight Face Feature Embedder (~2.0M parameters)"""

    def __init__(self, embedding_dim=128, num_classes=10000, input_size=112, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size

        # Backbone
        self.backbone = MobileNetV2Lite(in_channels=3, channels=[16, 24, 32, 96, 320], expand_ratio=4)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(320, 512),
            nn.BatchNorm1d(512),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # ArcFace head
        self.arcface = ArcFaceHead(embedding_dim, num_classes, margin=0.2, scale=30.0)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_embedding(self, x):
        """Extract face embedding (inference mode)"""
        features = self.backbone(x)[-1]  # Use last stage features
        pooled = self.pool(features).flatten(1)
        embedding = self.embedding(pooled)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x, labels=None):
        """
        x: [B, 3, H, W]
        labels: [B] (optional, for training with ArcFace)
        """
        embedding = self.extract_embedding(x)

        if labels is not None:
            # Training mode: return ArcFace logits
            return self.arcface(embedding, labels)

        # Inference mode: return embedding
        return embedding

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FaceRecognitionSystem(nn.Module):
    """Complete Face Recognition System (Detector + Embedder)"""

    def __init__(self, num_identities=10000, input_size=224, embedder_input_size=112):
        super().__init__()
        self.input_size = input_size
        self.embedder_input_size = embedder_input_size

        # Face detector
        self.detector = FaceDetector(num_classes=1, input_size=input_size)

        # Face embedder
        self.embedder = FaceEmbedder(
            embedding_dim=128,
            num_classes=num_identities,
            input_size=embedder_input_size,
            dropout=0.3
        )

    def detect_faces(self, x):
        """Detect faces in image"""
        return self.detector(x)

    def extract_features(self, face_crops):
        """Extract features from face crops"""
        return self.embedder(face_crops)

    def forward(self, x, labels=None):
        """Forward pass"""
        return self.embedder(x, labels)

    def count_parameters(self):
        det_params = self.detector.count_parameters()
        emb_params = self.embedder.count_parameters()
        return {
            'detector': det_params,
            'embedder': emb_params,
            'total': det_params + emb_params
        }


def build_face_detector(input_size=224):
    """Build face detector model"""
    model = FaceDetector(num_classes=1, input_size=input_size)
    return model


def build_face_embedder(embedding_dim=128, num_classes=10000):
    """Build face embedder model"""
    model = FaceEmbedder(embedding_dim=embedding_dim, num_classes=num_classes)
    return model


def build_face_recognition_system(num_identities=10000):
    """Build complete face recognition system"""
    model = FaceRecognitionSystem(num_identities=num_identities)
    return model


if __name__ == '__main__':
    # Test model sizes
    detector = build_face_detector()
    embedder = build_face_embedder(embedding_dim=128, num_classes=10000)

    print(f"Face Detector Parameters: {detector.count_parameters():,}")
    print(f"Face Embedder Parameters: {embedder.count_parameters():,}")
    print(f"Total Parameters: {detector.count_parameters() + embedder.count_parameters():,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    cls_scores, bbox_preds = detector(x)
    print(f"\nDetection output shapes:")
    for i, (cls, box) in enumerate(zip(cls_scores, bbox_preds)):
        print(f"  Scale {i}: cls={cls.shape}, box={box.shape}")

    x = torch.randn(2, 3, 112, 112)
    embedding = embedder.extract_embedding(x)
    print(f"\nEmbedding shape: {embedding.shape}")