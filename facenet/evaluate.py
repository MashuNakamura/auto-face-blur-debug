#!/usr/bin/env python3
"""
FACENET - BATCH IMAGE EVALUATION
---------------------------------------------------
Test dari folder dataset_uji/ dengan 2 subfolder:
- saya/         :  Expected "Me"
- orang_lain/   : Expected "Unknown"
"""

import os
import shutil
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG
# ============================================================
THRESHOLD = 0.38
INPUT_RES = (640, 480)
TESTING_DIR = "../testing/"
OUTPUT_DIR = "../output/facenet/"
PATH_YOLO_MODEL = "../model/model.pt"

# ============================================================
# SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

# Load models
print("Loading models...")
yolo_model = YOLO(PATH_YOLO_MODEL)
yolo_model.to(device)

facenet_model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
facenet_model.eval()
print(f"✓ Models loaded")

# Create output dir
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================
@torch.no_grad()
def get_embedding(img_bgr):
    """Extract FaceNet embedding"""
    if img_bgr is None or img_bgr.size == 0:
        return None

    inp = cv2.resize(img_bgr, (160, 160))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(inp).float().to(device)
    tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return facenet_model(tensor).flatten().cpu().numpy()

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
print("FACENET - BATCH EVALUATION FOR MAP AND CONFUSION MATRIX")
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
