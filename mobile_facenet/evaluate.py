#!/usr/bin/env python3
"""
MOBILEFACENET - BATCH IMAGE EVALUATION (FINAL)
---------------------------------------------------
Evaluasi MobileFaceNet + YOLO dari folder testing/ dengan subfolder:
- testing/saya/        : whitelist / "Me"
- testing/orang_lain/  : "Unknown"
Output:
- Confusion Matrix (plot + teks)
- Precision / Recall / F1 / Accuracy (plot + teks)
- mAP + Precision-Recall Curve (plot + teks)
- Visualisasi contoh (blur applied untuk Unknown)
- Dokumentasi parameter eksperimen
"""

import os
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from mfnet import MobileFacenet
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
THRESHOLD = 0.45
INPUT_RES = (640, 480)
TESTING_DIR = "../testing/"
OUTPUT_DIR = "../output/mobilefacenet/"
PATH_YOLO_MODEL = "../model/model.pt"
PATH_MFNET_CKPT = "../model/068.ckpt"  # <-- GANTI NAMA FILE!
VISUALIZE_WHITELIST = ["Noel_1.jpg", "Bram_1.jpg"]
VISUALIZE_UNKNOWN = ["9_Press_Conference_Press_Conference_9_141.jpg",
                     "9_Press_Conference_Press_Conference_9_258.jpg"]

# ============================================================
# SETUP DEVICE & MODEL
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

# Load YOLO
print("Loading YOLO model...")
yolo_model = YOLO(PATH_YOLO_MODEL)
yolo_model.to(device)
print(f"✓ YOLO loaded")

# Load MobileFaceNet
print("Loading MobileFaceNet model...")
mfnet_model = MobileFacenet()

if os.path.exists(PATH_MFNET_CKPT):
    checkpoint = torch.load(PATH_MFNET_CKPT, map_location='cpu')

    # Auto-detect checkpoint format
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'net_state_dict' in checkpoint:
            state_dict = checkpoint['net_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    mfnet_model.load_state_dict(state_dict, strict=False)
    print(f"✓ MobileFaceNet loaded")
else:
    print(f"ERROR:  Checkpoint not found")
    exit(1)

mfnet_model.to(device)
mfnet_model.eval()

# Prepare output folder
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================
@torch.no_grad()
def get_embedding(img_bgr):
    """Extract MobileFaceNet embedding"""
    if img_bgr is None or img_bgr.size == 0:
        return None
    try:
        inp = cv2.resize(img_bgr, (112, 112))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = (inp.astype(np.float32) - 127.5) / 128.0
        inp = np.transpose(inp, (2, 0, 1))
        tensor = torch.from_numpy(inp).float().unsqueeze(0).to(device)

        output = mfnet_model(tensor)
        if isinstance(output, tuple):
            emb = output[0]
        else:
            emb = output

        emb = emb.flatten().cpu().numpy()
        return emb / np.linalg.norm(emb)  # L2 normalize
    except:
        return None
    return None

def get_distance(emb1, emb2):
    """Cosine distance"""
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# ============================================================
# LOAD WHITELIST EMBEDDINGS
# ============================================================
print("\nLoading whitelist (Me)...")
whitelist_dir = os.path.join(TESTING_DIR, "saya")
target_embeddings = []

for fname in os.listdir(whitelist_dir):
    if fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
        path = os.path.join(whitelist_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        results = yolo_model(img, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
                face_crop = img[y1:y2, x1:x2]
                emb = get_embedding(face_crop)
                if emb is not None:
                    target_embeddings.append(emb)
                break

target_embeddings = np.array(target_embeddings)
print(f"Loaded {len(target_embeddings)} whitelist embeddings\n")

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_image(image_path, target_embeddings):
    img = cv2.imread(image_path)
    if img is None:
        return False, 1.0
    img = cv2.resize(img, INPUT_RES)
    results = yolo_model(img, verbose=False)
    for r in results:
        if len(r.boxes) == 0:
            continue
        x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        emb = get_embedding(face_crop)
        if emb is not None and len(target_embeddings)>0:
            dists = [get_distance(t, emb) for t in target_embeddings]
            score = min(dists)
            return score<=THRESHOLD, score
    return False, 1.0

# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
print("="*50)
print("MOBILEFACENET BATCH EVALUATION")
print("="*50)

y_true, y_pred, y_scores = [], [], []

# Process unknown
unknown_dir = os.path.join(TESTING_DIR, "orang_lain")
for fname in os.listdir(unknown_dir):
    if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
        continue
    path = os.path.join(unknown_dir, fname)
    is_known, score = evaluate_image(path, target_embeddings)
    y_true.append(0)
    y_pred.append(1 if is_known else 0)
    y_scores.append(1 - score)

# Process whitelist
for fname in os.listdir(whitelist_dir):
    if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
        continue
    path = os.path.join(whitelist_dir, fname)
    is_known, score = evaluate_image(path, target_embeddings)
    y_true.append(1)
    y_pred.append(1 if is_known else 0)
    y_scores.append(1 - score)

# ============================================================
# VISUALISASI CONTOH
# ============================================================
def save_visual_example(fname_list, folder, label_name, blur=False):
    for fname in fname_list:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        results = yolo_model(img, verbose=False)
        for r in results:
            if len(r.boxes) == 0:
                continue
            x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
            face_crop = img[y1:y2, x1:x2]
            if blur:
                try:
                    img[y1:y2, x1:x2] = cv2.GaussianBlur(face_crop, (151,151), 50)
                except:
                    pass
            color = (0,255,0) if not blur else (0,0,255)
            label = f"{label_name}"
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            out_name = f"example_{label_name}_{fname}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), img)
            break

save_visual_example(VISUALIZE_WHITELIST, whitelist_dir, "Whitelist", blur=False)
save_visual_example(VISUALIZE_UNKNOWN, unknown_dir, "Unknown", blur=True)

# ============================================================
# CALCULATE METRICS
# ============================================================
cm = confusion_matrix(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1])
accuracy = np.sum(np.array(y_true)==np.array(y_pred))/len(y_true)
map_score = average_precision_score(y_true, y_scores)

# Save results to txt
with open(os.path.join(OUTPUT_DIR,'evaluation_results.txt'),'w') as f:
    f.write("=== Evaluation Results ===\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\n")
    f.write(f"Precision: Unknown={precision[0]:.2f}, Me={precision[1]:.2f}\n")
    f.write(f"Recall:    Unknown={recall[0]:.2f}, Me={recall[1]:.2f}\n")
    f.write(f"F1-score:  Unknown={f1[0]:.2f}, Me={f1[1]:.2f}\n")
    f.write(f"Accuracy:  {accuracy:.2f}\n")
    f.write(f"Mean Average Precision (mAP): {map_score:.4f}\n")

print("\nConfusion Matrix:")
print(cm)
print(f"\nPrecision: Unknown={precision[0]:.2f}, Me={precision[1]:.2f}")
print(f"Recall:    Unknown={recall[0]:.2f}, Me={recall[1]:.2f}")
print(f"F1-score:  Unknown={f1[0]:.2f}, Me={f1[1]:.2f}")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Mean Average Precision (mAP): {map_score:.4f}")

# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unknown','Me'], yticklabels=['Unknown','Me'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR,'confusion_matrix.png'))
plt.close()

# ============================================================
# PLOT PRECISION-RECALL CURVE
# ============================================================
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall_vals, precision_vals)

plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals, label=f'PR curve (AUC={pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR,'precision_recall_curve.png'))
plt.close()

# ============================================================
# LOG PARAMETER EXPERIMENT
# ============================================================
with open(os.path.join(OUTPUT_DIR,'experiment_params.txt'),'w') as f:
    f.write("=== Experiment Parameters ===\n")
    f.write(f"Threshold: {THRESHOLD}\n")
    f.write(f"Input Resolution: {INPUT_RES}\n")
    f.write(f"Whitelist count: {len(target_embeddings)}\n")
    f.write(f"Unknown count: {len(os.listdir(unknown_dir))}\n")
    f.write(f"YOLO model path: {PATH_YOLO_MODEL}\n")
    f.write(f"MobileFaceNet checkpoint: {PATH_MFNET_CKPT}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Visualize whitelist examples: {VISUALIZE_WHITELIST}\n")
    f.write(f"Visualize unknown examples: {VISUALIZE_UNKNOWN}\n")

print(f"\n✓ All outputs saved to {OUTPUT_DIR}")
print("Done!")