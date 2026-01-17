#!/usr/bin/env python3
"""
AUTO WHITELIST FACE CAPTURE (MULTI-VIEW)
---------------------------------------
- YOLO Face Detection
- Auto capture many face angles
- Save to ./whitelist/<person_name>/
"""

import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================
YOLO_MODEL_PATH = "./model/model.pt"
WHITELIST_DIR = "./whitelist"
CAPTURE_INTERVAL = 10        # capture every N frames
MAX_IMAGES = 50              # stop after this many images
MIN_FACE_SIZE = 80           # px
INPUT_RES = (640, 480)

# ==============================
# SETUP
# ==============================
name = input("Nama whitelist (contoh: mashu): ").strip().lower()
assert name != ""

save_dir = os.path.join(WHITELIST_DIR, name)
os.makedirs(save_dir, exist_ok=True)

model = YOLO(YOLO_MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RES[1])

print(f"\n[INFO] Capture untuk: {name}")
print("[INFO] Gerakkan kepala perlahan (kiri / kanan / atas / bawah)")
print("[INFO] Tekan Q untuk stop manual\n")

frame_count = 0
saved_count = 0
last_saved_box = None

# ==============================
# HELPER
# ==============================
def box_distance(b1, b2):
    if b1 is None or b2 is None:
        return 9999
    c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
    c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
    return np.linalg.norm(np.array(c1) - np.array(c2))

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, INPUT_RES)
    display = frame.copy()
    frame_count += 1

    results = model(frame, verbose=False)
    for r in results:
        if len(r.boxes) == 0:
            continue

        box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        w, h = x2 - x1, y2 - y1
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2)

        if frame_count % CAPTURE_INTERVAL == 0:
            dist = box_distance(box, last_saved_box)
            if dist > 25:
                face = frame[y1:y2, x1:x2]
                fname = f"{name}_{saved_count:03d}.jpg"
                path = os.path.join(save_dir, fname)
                cv2.imwrite(path, face)
                saved_count += 1
                last_saved_box = box
                print(f"[SAVED] {fname}")

        break  # only first face

    cv2.putText(display, f"Saved: {saved_count}/{MAX_IMAGES}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Auto Whitelist Capture", display)

    if saved_count >= MAX_IMAGES:
        print("\n[INFO] Target tercapai")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n[DONE] Total tersimpan: {saved_count} gambar")
print(f"[DIR ] {save_dir}")