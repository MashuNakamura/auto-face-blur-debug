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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CONFIG
# ============================================================
THRESHOLD = 0.50
INPUT_RES = (640, 480)
WHITELIST_DIR = "../whitelist/"
DATASET_DIR = "../dataset_uji/"
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
# LOAD WHITELIST
# ============================================================
print("\nLoading whitelist...")
target_embeddings = []

for fname in os.listdir(WHITELIST_DIR):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '. bmp')):
        path = os.path.join(WHITELIST_DIR, fname)
        img = cv2.imread(path)
        if img is not None:
            results = yolo_model(img, verbose=False)
            for r in results:
                if len(r.boxes) > 0:
                    x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
                    face_crop = img[y1:y2, x1:x2]
                    emb = get_embedding(face_crop)
                    if emb is not None:
                        target_embeddings.append(emb)
                        print(f"  ✓ {fname}")
                    break

target_embeddings = np.array(target_embeddings)
print(f"Total: {len(target_embeddings)} embeddings\n")


# ============================================================
# PROCESS FUNCTION
# ============================================================
def process_folder(folder_name, expected_status):
    """Process saya/ or orang_lain/ folder"""
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_name} not found, skipping...")
        return

    print(f"\nProcessing:  {folder_name}/ (Expected: {expected_status})")

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    correct_count = 0
    total_count = 0

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, INPUT_RES)

        # Detect face
        results = yolo_model(img, verbose=False)

        for r in results:
            if len(r.boxes) == 0:
                continue

            box = r.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_img = img[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # Recognize
            is_known = False
            score = 1.0

            if len(target_embeddings) > 0:
                emb = get_embedding(face_img)
                if emb is not None:
                    dists = [get_distance(t, emb) for t in target_embeddings]
                    score = min(dists)
                    if score <= THRESHOLD:
                        is_known = True

            actual_status = "Me" if is_known else "Unknown"
            is_correct = (actual_status == expected_status)

            # Visualize
            canvas = img.copy()

            # Blur if unknown
            if not is_known:
                try:
                    canvas[y1:y2, x1:x2] = cv2.GaussianBlur(canvas[y1:y2, x1:x2], (51, 51), 30)
                except:
                    pass

            # Box
            color = (0, 255, 0) if is_known else (0, 0, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"{actual_status} ({score:.2f})"
            cv2.putText(canvas, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Model name
            cv2.rectangle(canvas, (0, 0), (200, 35), (0, 0, 0), -1)
            cv2.putText(canvas, "ArcFace", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Result
            result_text = "BENAR" if is_correct else "SALAH"
            result_color = (0, 255, 0) if is_correct else (0, 0, 255)

            cv2.rectangle(canvas, (0, 440), (150, 480), (0, 0, 0), -1)
            cv2.putText(canvas, result_text, (5, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 3)

            # Save
            output_path = os.path.join(OUTPUT_DIR, f"{folder_name}_{fname}")
            cv2.imwrite(output_path, canvas)

            if is_correct:
                correct_count += 1
            total_count += 1

            print(f"  {fname}: {actual_status} ({'✓' if is_correct else '✗'})")
            break

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"\nAccuracy: {correct_count}/{total_count} = {accuracy:.2f}%")


# ============================================================
# MAIN
# ============================================================
print("=" * 50)
print("ARCFACE - BATCH EVALUATION")
print("=" * 50)

process_folder("saya", "Me")
process_folder("orang_lain", "Unknown")

print(f"\n✓ Results saved to:  {OUTPUT_DIR}")
print("Done!")