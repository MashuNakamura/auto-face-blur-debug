#!/usr/bin/env python3
"""
YOLO BACKBONE - BATCH IMAGE EVALUATION
---------------------------------------------------
Test dari folder dataset_uji/ dengan 2 subfolder:
- saya/         : Expected "Me"
- orang_lain/   : Expected "Unknown"
"""

import os
import shutil
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
THRESHOLD = 0.35
INPUT_RES = (640, 480)
WHITELIST_DIR = "../whitelist/"
DATASET_DIR = "../dataset_uji/"
OUTPUT_DIR = "../output/yolo_backbone/"
PATH_YOLO_MODEL = "../model/model.pt"

# ============================================================
# SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

# Load model
print("Loading YOLO model...")
model = YOLO(PATH_YOLO_MODEL)
model.to(device)
print(f"✓ Model loaded")

# Create output dir
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================
@torch.no_grad()
def get_embedding(img_bgr):
    """Extract YOLO backbone embedding"""
    if img_bgr is None or img_bgr.size == 0:
        return None

    inp = cv2.resize(img_bgr, (128, 128))
    tensor = torch.from_numpy(inp).float().to(device) / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    x = tensor
    for i, layer in enumerate(model.model.model):
        x = layer(x)
        if i == 9:
            break

    emb = torch.mean(x, dim=(2, 3)).flatten()
    return (emb / emb.norm()).cpu().numpy()


def get_distance(emb1, emb2):
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# ============================================================
# LOAD WHITELIST
# ============================================================
print("\nLoading whitelist...")
target_embeddings = []

for fname in os.listdir(WHITELIST_DIR):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        path = os.path.join(WHITELIST_DIR, fname)
        img = cv2.imread(path)
        if img is not None:
            results = model(img, verbose=False)
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
        results = model(img, verbose=False)

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
            cv2.putText(canvas, "YOLO-Backbone", (5, 25),
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
print("YOLO BACKBONE - BATCH EVALUATION")
print("=" * 50)

process_folder("saya", "Me")
process_folder("orang_lain", "Unknown")

print(f"\n✓ Results saved to:  {OUTPUT_DIR}")
print("Done!")