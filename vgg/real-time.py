#!/usr/bin/env python3
"""
VGG-FACE - REALTIME WEBCAM TEST
---------------------------------------------------
Detektor    : YOLOv11n
Vektorisasi : VGG-Face (via DeepFace)
Threshold   : 0.40
"""

import os
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
THRESHOLD = 0.40
INPUT_RES = (640, 480)
WHITELIST_DIR = "../whitelist/"
PATH_YOLO_MODEL = "../model/model.pt"
COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)

# ============================================================
# SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device: {device}")

# Select camera
CAM_INDEX = int(input("Pilih kamera (0,1,2,... ): ") or 0)

# Load YOLO
print("Loading YOLO model...")
yolo_model = YOLO(PATH_YOLO_MODEL)
yolo_model.to(device)
print(f"✓ YOLO loaded on {device}")

# Pre-load VGG-Face model
print("Pre-loading VGG-Face model...")
dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
try:
    DeepFace.represent(
        img_path=dummy,
        model_name="VGG-Face",
        enforce_detection=False,
        detector_backend="skip"
    )
    print("✓ VGG-Face model loaded")
except Exception as e:
    print(f"Warning: VGG-Face pre-load failed: {e}")


# ============================================================
# EMBEDDING FUNCTION
# ============================================================
def get_embedding(img_bgr):
    """Extract VGG-Face embedding"""
    if img_bgr is None or img_bgr.size == 0:
        return None

    try:
        # Convert to RGB
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # Ensure proper size (VGG needs at least 224x224)
        h, w = rgb.shape[:2]
        if h < 224 or w < 224:
            scale = 224 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            rgb = cv2.resize(rgb, (new_w, new_h))

        # Extract embedding
        result = DeepFace.represent(
            img_path=rgb,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="skip"
        )

        if result and len(result) > 0 and "embedding" in result[0]:
            embedding = np.array(result[0]["embedding"])
            return embedding / np.linalg.norm(embedding)

    except Exception as e:
        # Silent fail untuk realtime
        pass

    return None


def get_distance(emb1, emb2):
    """Cosine distance"""
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def blur_face(face_img):
    """Gaussian blur"""
    return cv2.GaussianBlur(face_img, (51, 51), 30)


# ============================================================
# LOAD WHITELIST
# ============================================================
print("\nLoading whitelist...")
target_embeddings = []

if not os.path.exists(WHITELIST_DIR):
    os.makedirs(WHITELIST_DIR)
    print("WARNING: Whitelist folder empty")

for fname in os.listdir(WHITELIST_DIR):
    if fname.lower().endswith(('.jpg', '. jpeg', '.png', '.bmp')):
        path = os.path.join(WHITELIST_DIR, fname)
        img = cv2.imread(path)
        if img is not None:
            # Detect face first
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
print(f"Total:  {len(target_embeddings)} embeddings loaded\n")

# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RES[1])

prev_time = time.time()

print("=" * 50)
print("VGG-FACE - REALTIME TEST")
print("Controls:  [q] Quit | [t] Toggle CPU/GPU")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sframe = cv2.resize(frame, INPUT_RES)
    display = sframe.copy()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Inference time
    t_start = time.time()

    # Detect faces
    results = yolo_model(sframe, stream=True, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = sframe.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_img = sframe[y1:y2, x1:x2]
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

            # Visualize
            if is_known:
                color = COLOR_KNOWN
                label = f"Me ({score:.2f})"
            else:
                color = COLOR_UNKNOWN
                label = f"Unknown ({score:.2f})"
                try:
                    display[y1:y2, x1:x2] = blur_face(display[y1:y2, x1:x2])
                except:
                    pass

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    t_end = time.time()
    inference_ms = (t_end - t_start) * 1000

    # UI Info
    dev_str = "GPU" if device.type == "cuda" else "CPU"
    info = f"[{dev_str}] VGG-Face | Infer: {inference_ms:. 1f}ms | FPS:  {fps:.1f}"

    cv2.rectangle(display, (0, 0), (640, 35), (0, 0, 0), -1)
    cv2.putText(display, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("VGG-Face - Realtime", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        # Toggle device (YOLO only, VGG stays on CPU)
        new_dev = "cpu" if device.type == "cuda" else "cuda"
        if new_dev == "cuda" and not torch.cuda.is_available():
            print("CUDA not available!")
            continue

        print(f"\nSwitching YOLO to {new_dev.upper()}...")
        device = torch.device(new_dev)
        yolo_model.to(device)
        print("Done!")

cap.release()
cv2.destroyAllWindows()