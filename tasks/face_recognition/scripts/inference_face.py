"""
Face Recognition Inference and Registration System
Supports face detection, feature extraction, and face matching
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import pickle
from datetime import datetime


class FaceDetector:
    """Face detection wrapper"""

    def __init__(self, model_path, input_size=224, device='cpu'):
        self.input_size = input_size
        self.device = torch.device(device)

        # Load model
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from face_models import FaceDetector as FaceDetectorModel

        self.model = FaceDetectorModel(num_classes=1, input_size=input_size)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded detector from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found, using untrained model")

        self.model.to(self.device)
        self.model.eval()

    def detect(self, img, score_threshold=0.5, nms_threshold=0.4):
        """
        Detect faces in image
        Returns: List of [x, y, w, h, score]
        """
        h, w = img.shape[:2]

        # Preprocess
        input_img = cv2.resize(img, (self.input_size, self.input_size))
        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0
        input_tensor = (input_tensor - 0.5) / 0.5
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Detect
        with torch.no_grad():
            cls_scores, bbox_preds = self.model(input_tensor)

        # Post-process
        detections = []
        scale_factor = max(h, w) / self.input_size

        for cls_score, box_pred in zip(cls_scores, bbox_preds):
            # Apply sigmoid to get probability
            cls_prob = torch.sigmoid(cls_score)[0]  # [1, H, W]

            # Get positive predictions
            pos_mask = cls_prob > score_threshold

            if pos_mask.sum() > 0:
                # Get indices
                indices = pos_mask.nonzero(as_tuple=False)

                for idx in indices:
                    cy, cx = idx[0].item(), idx[1].item()
                    score = cls_prob[0, cy, cx].item()

                    # Decode box
                    dx = box_pred[0, cy, cx].item()
                    dy = box_pred[1, cy, cx].item()
                    dw = box_pred[2, cy, cx].item()
                    dh = box_pred[3, cy, cx].item()

                    # Convert to image coordinates
                    stride = self.input_size / cls_prob.shape[1]
                    cx_scaled = (cx + dx) * stride
                    cy_scaled = (cy + dy) * stride
                    w_scaled = np.exp(dw) * stride * 4  # Scale factor for anchor
                    h_scaled = np.exp(dh) * stride * 4

                    # Convert to [x, y, w, h]
                    x = int((cx_scaled - w_scaled / 2) * scale_factor)
                    y = int((cy_scaled - h_scaled / 2) * scale_factor)
                    w = int(w_scaled * scale_factor)
                    h = int(h_scaled * scale_factor)

                    # Clip to image bounds
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    w = min(w, w - x)
                    h = min(h, h - y)

                    detections.append([x, y, w, h, score])

        # Apply NMS
        if len(detections) > 0:
            detections = self._nms(detections, nms_threshold)

        return detections

    def _nms(self, detections, nms_threshold):
        """Non-maximum suppression"""
        if len(detections) == 0:
            return []

        # Sort by score
        detections = sorted(detections, key=lambda x: x[4], reverse=True)

        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)

            # Remove overlapping boxes
            remaining = []
            for det in detections:
                if self._iou(current[:4], det[:4]) < nms_threshold:
                    remaining.append(det)

            detections = remaining

        return keep

    def _iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2

        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class FaceEmbedder:
    """Face feature extraction wrapper"""

    def __init__(self, model_path, embedding_dim=128, input_size=112, device='cpu'):
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.device = torch.device(device)

        # Load model
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from face_models import FaceEmbedder as FaceEmbedderModel

        self.model = FaceEmbedderModel(embedding_dim=embedding_dim, num_classes=10000, input_size=input_size)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded embedder from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found, using untrained model")

        # Load label mapping if available
        label_mapping_path = os.path.join(os.path.dirname(model_path), "label_mapping.json")
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                self.label_mapping = json.load(f)
        else:
            self.label_mapping = None

        self.model.to(self.device)
        self.model.eval()

    def extract(self, face_img):
        """
        Extract feature from face crop
        Returns: Normalized embedding vector
        """
        # Preprocess
        if face_img.shape[0] != self.input_size or face_img.shape[1] != self.input_size:
            face_img = cv2.resize(face_img, (self.input_size, self.input_size))

        input_tensor = torch.from_numpy(face_img.transpose(2, 0, 1)).float() / 255.0
        input_tensor = (input_tensor - 0.5) / 0.5
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            embedding = self.model.extract_embedding(input_tensor)

        return embedding.cpu().numpy()[0]


class FaceDatabase:
    """Face registration database"""

    def __init__(self, database_path=None):
        self.database_path = database_path or "face_database.pkl"
        self.embeddings = []
        self.metadata = []
        self._load()

    def _load(self):
        """Load database from file"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data.get('embeddings', [])
                self.metadata = data.get('metadata', [])
            print(f"Loaded {len(self.embeddings)} registered faces")
        else:
            print("No existing database found, starting fresh")

    def save(self):
        """Save database to file"""
        with open(self.database_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }, f)
        print(f"Saved database with {len(self.embeddings)} faces")

    def register(self, embedding, name, metadata=None):
        """Register a new face"""
        self.embeddings.append(embedding)
        meta = {
            'name': name,
            'id': len(self.embeddings) - 1,
            'timestamp': datetime.now().isoformat(),
            **(metadata or {})
        }
        self.metadata.append(meta)
        print(f"Registered face: {name} (ID: {meta['id']})")
        return meta['id']

    def search(self, embedding, threshold=0.5, top_k=5):
        """
        Search for matching faces
        Returns: List of (metadata, similarity_score)
        """
        if len(self.embeddings) == 0:
            return []

        # Calculate similarities
        embeddings_matrix = np.array(self.embeddings)
        embedding = embedding.reshape(1, -1)

        # Cosine similarity
        similarities = np.dot(embeddings_matrix, embedding) / (
            np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(embedding) + 1e-8
        )

        # Get top-k matches
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append((self.metadata[idx], similarities[idx]))

        return results


class FaceRecognitionSystem:
    """Complete face recognition system"""

    def __init__(self, detector_path=None, embedder_path=None, database_path=None,
                 input_size=224, embedder_input_size=112, embedding_dim=128, device='cpu'):
        self.device = torch.device(device)

        # Initialize detector
        if detector_path:
            self.detector = FaceDetector(detector_path, input_size, device)
        else:
            self.detector = None

        # Initialize embedder
        if embedder_path:
            self.embedder = FaceEmbedder(embedder_path, embedding_dim, embedder_input_size, device)
        else:
            self.embedder = None

        # Initialize database
        self.database = FaceDatabase(database_path)

    def detect_faces(self, img, draw=True):
        """Detect all faces in image"""
        if self.detector is None:
            raise ValueError("Detector not initialized")

        detections = self.detector.detect(img)

        if draw:
            result_img = img.copy()
            for i, (x, y, w, h, score) in enumerate(detections):
                # Draw bounding box
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw label
                label = f"Face {i+1}: {score:.2f}"
                cv2.putText(result_img, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return detections, result_img

        return detections

    def register_face(self, img, name, metadata=None):
        """Detect and register a face"""
        if self.detector is None or self.embedder is None:
            raise ValueError("Detector or embedder not initialized")

        # Detect faces
        detections = self.detector.detect(img)

        if len(detections) == 0:
            print("No face detected!")
            return None
        elif len(detections) > 1:
            print(f"Warning: Multiple faces detected ({len(detections)}), using the first one")

        # Get the largest face
        detections = sorted(detections, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h, score = detections[0]

        # Extract face crop with margin
        margin = int(max(w, h) * 0.2)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)

        face_crop = img[y1:y2, x1:x2]

        # Extract features
        embedding = self.embedder.extract(face_crop)

        # Register to database
        face_id = self.database.register(embedding, name, metadata)

        # Draw result
        result_img = img.copy()
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_img, f"Registered: {name}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return face_id, result_img

    def verify_face(self, img, threshold=0.5):
        """Verify face against registered database"""
        if self.detector is None or self.embedder is None:
            raise ValueError("Detector or embedder not initialized")

        # Detect faces
        detections = self.detector.detect(img)

        if len(detections) == 0:
            print("No face detected!")
            return None, None, img

        if len(detections) > 1:
            print(f"Warning: Multiple faces detected ({len(detections)}), using the first one")

        # Get the largest face
        detections = sorted(detections, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h, score = detections[0]

        # Extract face crop
        margin = int(max(w, h) * 0.2)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)

        face_crop = img[y1:y2, x1:x2]

        # Extract features
        embedding = self.embedder.extract(face_crop)

        # Search database
        results = self.database.search(embedding, threshold=threshold, top_k=1)

        # Draw result
        result_img = img.copy()
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if results:
            metadata, similarity = results[0]
            label = f"{metadata['name']}: {similarity:.2f}"
            color = (0, 255, 0) if similarity >= threshold else (0, 0, 255)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.putText(result_img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return results[0] if results else None, embedding, result_img


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Inference")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["detect", "register", "verify", "interactive"],
                       help="Operation mode")
    parser.add_argument("--detector", type=str, default="runs/face_recognition/detector/best_detector.pth",
                       help="Detector model path")
    parser.add_argument("--embedder", type=str, default="runs/face_recognition/embedder/best_embedder.pth",
                       help="Embedder model path")
    parser.add_argument("--database", type=str, default="face_database.pkl",
                       help="Face database path")
    parser.add_argument("--input", type=str, required=True,
                       help="Input image or directory")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory")
    parser.add_argument("--name", type=str, help="Name for face registration")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Verification threshold")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize system
    print("Initializing Face Recognition System...")
    face_system = FaceRecognitionSystem(
        detector_path=args.detector if os.path.exists(args.detector) else None,
        embedder_path=args.embedder if os.path.exists(args.embedder) else None,
        database_path=args.database,
        device=args.device
    )

    # Process based on mode
    if args.mode == "detect":
        # Single image detection
        img = cv2.imread(args.input)
        if img is None:
            print(f"Error: Could not read image {args.input}")
            return

        detections, result_img = face_system.detect_faces(img)

        output_path = os.path.join(args.output, f"detection_{Path(args.input).name}")
        cv2.imwrite(output_path, result_img)
        print(f"Saved result to {output_path}")
        print(f"Detected {len(detections)} faces")

    elif args.mode == "register":
        # Register a face
        img = cv2.imread(args.input)
        if img is None:
            print(f"Error: Could not read image {args.input}")
            return

        if not args.name:
            print("Error: --name is required for registration mode")
            return

        face_id, result_img = face_system.register_face(img, args.name)
        face_system.database.save()

        output_path = os.path.join(args.output, f"registered_{args.name}_{Path(args.input).name}")
        cv2.imwrite(output_path, result_img)
        print(f"Saved result to {output_path}")

    elif args.mode == "verify":
        # Verify a face
        img = cv2.imread(args.input)
        if img is None:
            print(f"Error: Could not read image {args.input}")
            return

        result, embedding, result_img = face_system.verify_face(img, args.threshold)

        output_path = os.path.join(args.output, f"verification_{Path(args.input).name}")
        cv2.imwrite(output_path, result_img)
        print(f"Saved result to {output_path}")

        if result:
            metadata, similarity = result
            print(f"Verified: {metadata['name']} (similarity: {similarity:.4f})")
        else:
            print("Face not recognized")

    elif args.mode == "interactive":
        # Interactive mode (webcam or image file selection)
        print("Interactive mode - use --input to specify image file")
        print("Use --mode detect/register/verify for batch processing")


if __name__ == "__main__":
    main()