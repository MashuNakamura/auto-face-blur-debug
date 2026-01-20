#!/usr/bin/env python3
"""
ARCFACE - BATCH IMAGE EVALUATION
---------------------------------------------------
Test dari folder dataset_uji/ dengan 2 subfolder:
- saya/         :  Expected "Me"
- orang_lain/   : Expected "Unknown"
"""

import os
import shutil
import time
import warnings
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics import confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CONFIG
# ============================================================
THRESHOLD = 0.50
INPUT_RES = (640, 480)
TESTING_DIR = "../testing/"
OUTPUT_DIR = "../output/arcface/"
PATH_YOLO_MODEL = "../model/model.pt"

# ============================================================
# SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

# Load YOLO
print("Loading YOLO model...")
yolo_model = YOLO(PATH_YOLO_MODEL)
yolo_model.to(device)
print(f"✓ YOLO loaded")

# Pre-load ArcFace
print("Pre-loading ArcFace model...")
dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
try:
    DeepFace.represent(
        img_path=dummy,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="skip"
    )
    print("✓ ArcFace model loaded")
except Exception as e:
    print(f"Warning:  {e}")

# Create output dir
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================
def get_embedding(img_bgr):
    """Extract ArcFace embedding"""
    if img_bgr is None or img_bgr.size == 0:
        return None

    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        h, w = rgb.shape[:2]
        if h < 224 or w < 224:
            scale = 224 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            rgb = cv2.resize(rgb, (new_w, new_h))
        elif h > 512 or w > 512:
            scale = 512 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            rgb = cv2.resize(rgb, (new_w, new_h))

        result = DeepFace.represent(
            img_path=rgb,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip"
        )

        if result and len(result) > 0 and "embedding" in result[0]:
            embedding = np.array(result[0]["embedding"])
            return embedding / np.linalg.norm(embedding)

    except Exception as e:
        print(f"  Embedding error: {str(e)[:50]}...")

    return None


def get_distance(emb1, emb2):
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# ============================================================
# PROCESS FUNCTION
# ============================================================
def evaluate_image(image_path, target_embeddings):
    """Process a single image and return if it's known and the score."""
    img = cv2.imread(image_path)
    if img is None:
        return False, 1.0

    img = cv2.resize(img, INPUT_RES)
    results = yolo_model(img, verbose=False)

    for r in results:
        if len(r.boxes) > 0:
            box = r.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_img = img[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            emb = get_embedding(face_img)
            if emb is not None:
                if len(target_embeddings) > 0:
                    dists = [get_distance(t, emb) for t in target_embeddings]
                    score = min(dists)
                    if score <= THRESHOLD:
                        return True, score
                else:
                    return False, 1.0

    return False, 1.0

# ============================================================
# MAIN
# ============================================================
print("=" * 50)
print("ARCFACE - BATCH EVALUATION FOR MAP AND CONFUSION MATRIX")
print("=" * 50)

y_true = []
y_pred = []
y_scores = []

image_files = [f for f in os.listdir(TESTING_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

for image_file in image_files:
    image_path = os.path.join(TESTING_DIR, image_file)
    print(f"Processing {image_file}...")

    # 1. Test without whitelist (expected: Unknown/not-blurred)
    is_known, score = evaluate_image(image_path, [])
    y_true.append(0) # 0 for Unknown
    y_pred.append(1 if is_known else 0)
    y_scores.append(1 - score)

    # 2. Test with whitelist (expected: Me/blurred)
    img = cv2.imread(image_path)
    if img is not None:
        results = yolo_model(img, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
                face_crop = img[y1:y2, x1:x2]
                emb = get_embedding(face_crop)
                if emb is not None:
                    is_known, score = evaluate_image(image_path, [emb])
                    y_true.append(1) # 1 for Me
                    y_pred.append(1 if is_known else 0)
                    y_scores.append(1 - score)
                    break

# Calculate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unknown', 'Me'], yticklabels=['Unknown', 'Me'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
print(f"\n✓ Confusion matrix plot saved to: {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")


# Calculate mAP
map_score = average_precision_score(y_true, y_scores)
print(f"\nMean Average Precision (mAP): {map_score:.4f}")


print("\nDone!")
