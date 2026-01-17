# #!/usr/bin/env python3
# """
# MOBILEFACENET - REALTIME WEBCAM TEST (GPU MODE ONLY)
# ---------------------------------------------------
# Detektor    : YOLOv11n (GPU)
# Vektorisasi : MobileFaceNet (GPU)
# Fitur       : Auto-Blur Unknown, Real-time FPS, Face Tracking, Embedding Cache
# """
#
# import os
# import time
# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO
# from mfnet import MobileFacenet
#
# # ============================================================
# # CONFIG
# # ============================================================
# THRESHOLD = 0.45
# INPUT_RES = (640, 480)
# WHITELIST_DIR = "../whitelist/"
# PATH_YOLO_MODEL = "../model/model.pt"
# PATH_MFNET_CKPT = "../model/068.ckpt"
# COLOR_KNOWN = (0, 255, 0)
# COLOR_UNKNOWN = (0, 0, 255)
#
# YOLO_SKIP_FRAMES = 2
# EMBEDDING_SKIP_FRAMES = 20
# TRACKING_DISTANCE_THRESHOLD = 60
# CACHE_MAX_AGE = 8
#
# fps_history = []
# max_fps_samples = 300
#
# # ============================================================
# # SETUP
# # ============================================================
# device = torch.device("cuda")
# print(f"[*] MODE: GPU ONLY")
# print(f"[*] Device: {device}")
# print(f"[*] CUDA Available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("[!] ERROR: CUDA not available!")
#     exit(1)
#
# CAM_INDEX = int(input("Pilih kamera (0,1,2,... ): ") or 0)
#
# # Load YOLO
# print("Loading YOLO model...")
# yolo_model = YOLO(PATH_YOLO_MODEL)
# yolo_model.to(device)
# print(f"✓ YOLO loaded on GPU")
#
# # Load MobileFaceNet
# print("Loading MobileFaceNet model...")
# mfnet_model = MobileFacenet()
#
# if os.path.exists(PATH_MFNET_CKPT):
#     checkpoint = torch.load(PATH_MFNET_CKPT, map_location='cpu')
#     if 'net_state_dict' in checkpoint:
#         mfnet_model. load_state_dict(checkpoint['net_state_dict'], strict=False)
#     else:
#         mfnet_model.load_state_dict(checkpoint, strict=False)
#     print(f"✓ MobileFaceNet loaded from checkpoint")
# else:
#     print(f"ERROR:  Checkpoint not found:  {PATH_MFNET_CKPT}")
#     exit(1)
#
# mfnet_model.to(device)
# mfnet_model.eval()
#
# # ============================================================
# # EMBEDDING CACHE SYSTEM
# # ============================================================
# embedding_cache = {}
#
# def get_box_key(box):
#     x1 = int(box[0] / 10) * 10
#     y1 = int(box[1] / 10) * 10
#     x2 = int(box[2] / 10) * 10
#     y2 = int(box[3] / 10) * 10
#     return f"{x1}_{y1}_{x2}_{y2}"
#
# def get_cached_embedding(box, current_time):
#     box_key = get_box_key(box)
#     if box_key in embedding_cache:
#         cache_entry = embedding_cache[box_key]
#         age = current_time - cache_entry['timestamp']
#         if age < CACHE_MAX_AGE:
#             return cache_entry['embedding']
#         else:
#             del embedding_cache[box_key]
#     return None
#
# def cache_embedding(box, embedding, current_time):
#     box_key = get_box_key(box)
#     embedding_cache[box_key] = {
#         'embedding': embedding,
#         'timestamp': current_time
#     }
#
# def clean_old_cache(current_time):
#     expired_keys = []
#     for key, entry in embedding_cache.items():
#         if current_time - entry['timestamp'] > CACHE_MAX_AGE:
#             expired_keys.append(key)
#     for key in expired_keys:
#         del embedding_cache[key]
#
# # ============================================================
# # EMBEDDING FUNCTION
# # ============================================================
# @torch.no_grad()
# def get_embedding(img_bgr):
#     if img_bgr is None or img_bgr.size == 0:
#         return None
#     try:
#         inp = cv2.resize(img_bgr, (112, 112))
#         inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
#         inp = (inp. astype(np.float32) - 127.5) / 128.0
#         inp = np.transpose(inp, (2, 0, 1))
#         tensor = torch.from_numpy(inp).float().unsqueeze(0).to(device)
#         output = mfnet_model(tensor)
#         if isinstance(output, tuple):
#             emb = output[0]
#         else:
#             emb = output
#         emb = emb.flatten().cpu().numpy()
#         return emb / np.linalg.norm(emb)
#     except Exception:
#         pass
#     return None
#
# def get_distance(emb1, emb2):
#     return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
#
# def blur_face(face_img):
#     return cv2.GaussianBlur(face_img, (51, 51), 30)
#
# def calculate_box_distance(box1, box2):
#     x1_center = (box1[0] + box1[2]) / 2
#     y1_center = (box1[1] + box1[3]) / 2
#     x2_center = (box2[0] + box2[2]) / 2
#     y2_center = (box2[1] + box2[3]) / 2
#     distance = ((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2) ** 0.5
#     return distance
#
# def match_faces_to_previous(new_detections, previous_faces):
#     matched_faces = []
#     used_detections = set()
#     for prev_face in previous_faces:
#         best_match_idx = -1
#         best_distance = float('inf')
#         for i, new_detection in enumerate(new_detections):
#             if i in used_detections:
#                 continue
#             distance = calculate_box_distance(prev_face['box'], new_detection)
#             if distance < TRACKING_DISTANCE_THRESHOLD and distance < best_distance:
#                 best_distance = distance
#                 best_match_idx = i
#         if best_match_idx != -1:
#             updated_face = prev_face.copy()
#             updated_face['box'] = new_detections[best_match_idx]
#             matched_faces.append(updated_face)
#             used_detections.add(best_match_idx)
#     for i, new_detection in enumerate(new_detections):
#         if i not in used_detections:
#             matched_faces.append({
#                 'box': new_detection,
#                 'status': False,
#                 'score': 1.0,
#                 'needs_recognition': True
#             })
#     return matched_faces
#
# # ============================================================
# # LOAD WHITELIST
# # ============================================================
# print("\nLoading whitelist...")
# target_embeddings = []
#
# if not os.path.exists(WHITELIST_DIR):
#     os.makedirs(WHITELIST_DIR)
#     print("WARNING: Whitelist folder empty")
#
# for fname in os.listdir(WHITELIST_DIR):
#     if fname.lower().endswith(('.jpg', '.jpeg', '.png', '. bmp')):
#         path = os.path.join(WHITELIST_DIR, fname)
#         img = cv2.imread(path)
#         if img is not None:
#             results = yolo_model(img, verbose=False)
#             for r in results:
#                 if len(r.boxes) > 0:
#                     x1, y1, x2, y2 = map(int, r.boxes[0].xyxy[0])
#                     face_crop = img[y1:y2, x1:x2]
#                     emb = get_embedding(face_crop)
#                     if emb is not None:
#                         target_embeddings.append(emb)
#                         print(f"  ✓ {fname}")
#                     break
#
# target_embeddings = np.array(target_embeddings)
# print(f"Total:  {len(target_embeddings)} embeddings loaded\n")
#
# # ============================================================
# # MAIN LOOP
# # ============================================================
# cap = cv2.VideoCapture(CAM_INDEX)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RES[0])
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RES[1])
#
# prev_time = time.time()
# frame_count = 0
# cache_hits = 0
# cache_misses = 0
# current_faces = []
#
# print("=" * 50)
# print("MOBILEFACENET - GPU MODE")
# print("=" * 50)
# print(" KONFIGURASI:")
# print(f" YOLO Detection:  Setiap {YOLO_SKIP_FRAMES} frame")
# print(f" Face Recognition: Setiap {EMBEDDING_SKIP_FRAMES} frame")
# print(f" Tracking Distance: {TRACKING_DISTANCE_THRESHOLD} pixels")
# print(f" Cache Max Age: {CACHE_MAX_AGE} seconds")
# print("")
# print(" KONTROL:")
# print(" [1] :  Set YOLO setiap 1 frame")
# print(" [5] : Set YOLO setiap 5 frame")
# print(" [0] : Set YOLO setiap 10 frame")
# print(" [c] : Clear embedding cache")
# print(" [q] : Keluar")
# print("=" * 50 + "\n")
#
# while True:
#     ret, frame = cap. read()
#     if not ret:
#         break
#
#     sframe = cv2.resize(frame, INPUT_RES)
#     display_frame = sframe.copy()
#
#     curr_time = time.time()
#     fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
#     prev_time = curr_time
#
#     fps_history.append(fps)
#     if len(fps_history) > max_fps_samples:
#         fps_history.pop(0)
#
#     frame_count += 1
#
#     if frame_count % 100 == 0:
#         clean_old_cache(curr_time)
#
#     yolo_status = "CACHED"
#     if frame_count % YOLO_SKIP_FRAMES == 0 or frame_count == 1:
#         yolo_status = "DETECTING"
#         new_detections = []
#         results = yolo_model(sframe, stream=True, verbose=False)
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 h, w = sframe.shape[:2]
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
#                 if (x2 - x1) > 10 and (y2 - y1) > 10:
#                     new_detections.append([x1, y1, x2, y2])
#         current_faces = match_faces_to_previous(new_detections, current_faces)
#
#     embedding_status = "CACHED"
#     if frame_count % EMBEDDING_SKIP_FRAMES == 0 or frame_count == 1:
#         embedding_status = "RECOGNIZING"
#         for face in current_faces:
#             x1, y1, x2, y2 = face['box']
#             face_img = sframe[y1:y2, x1:x2]
#             if face_img. size == 0:
#                 continue
#             is_whitelisted = False
#             score = 1.0
#             if len(target_embeddings) > 0:
#                 emb = get_cached_embedding(face['box'], curr_time)
#                 if emb is not None:
#                     cache_hits += 1
#                 else:
#                     cache_misses += 1
#                     emb = get_embedding(face_img)
#                     if emb is not None:
#                         cache_embedding(face['box'], emb, curr_time)
#                 if emb is not None:
#                     dists = [get_distance(t, emb) for t in target_embeddings]
#                     score = min(dists)
#                     if score <= THRESHOLD:
#                         is_whitelisted = True
#             face['status'] = is_whitelisted
#             face['score'] = score
#             face['needs_recognition'] = False
#
#     for face in current_faces:
#         x1, y1, x2, y2 = face['box']
#         if face['status']:
#             color = COLOR_KNOWN
#             label = f"Me ({face['score']:.2f})"
#         else:
#             color = COLOR_UNKNOWN
#             label = f"Unknown ({face['score']:.2f})"
#             try:
#                 display_frame[y1:y2, x1:x2] = blur_face(display_frame[y1:y2, x1:x2])
#             except:
#                 pass
#         cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(display_frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#     cache_ratio = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
#     avg_fps = np.mean(fps_history) if fps_history else 0
#
#     info_str = f"[GPU] MobileFaceNet | FPS: {fps:.1f} (Avg: {avg_fps:.1f}) | Cache:  {len(embedding_cache)}"
#     timing_str = f"YOLO: {yolo_status} | EMBED: {embedding_status} | Hit Rate: {cache_ratio:.1%}"
#
#     cv2.rectangle(display_frame, (0, 0), (640, 55), (0, 0, 0), -1)
#     cv2.putText(display_frame, info_str, (10, 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
#     cv2.putText(display_frame, timing_str, (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
#
#     cv2.imshow("MobileFaceNet - GPU Mode", display_frame)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('1'):
#         YOLO_SKIP_FRAMES = 1
#         print(f"\n--- YOLO Timing:  Setiap {YOLO_SKIP_FRAMES} frame ---")
#     elif key == ord('5'):
#         YOLO_SKIP_FRAMES = 5
#         print(f"\n--- YOLO Timing: Setiap {YOLO_SKIP_FRAMES} frame ---")
#     elif key == ord('0'):
#         YOLO_SKIP_FRAMES = 10
#         print(f"\n--- YOLO Timing: Setiap {YOLO_SKIP_FRAMES} frame ---")
#     elif key == ord('c'):
#         embedding_cache.clear()
#         cache_hits = 0
#         cache_misses = 0
#         print("\n--- Embedding cache cleared ---")
#
# cap.release()
# cv2.destroyAllWindows()
#
# # ============================================================
# # PERFORMANCE SUMMARY
# # ============================================================
# print("\n" + "=" * 60)
# print("PERFORMANCE SUMMARY - MOBILEFACENET GPU MODE")
# print("=" * 60)
#
# avg_fps = np.mean(fps_history) if fps_history else 0
# cache_hit_rate = (cache_hits / (cache_hits + cache_misses) * 100) if (cache_hits + cache_misses) > 0 else 0
#
# print(f"\n{'Metric':<25} {'Value':<20}")
# print("-" * 60)
# print(f"{'Model':<25} {'MobileFaceNet':<20}")
# print(f"{'Device':<25} {'GPU': <20}")
# print(f"{'Average FPS':<25} {avg_fps:<20.1f}")
# print(f"{'YOLO Skip Frames':<25} {YOLO_SKIP_FRAMES:<20}")
# print(f"{'Embedding Skip Frames':<25} {EMBEDDING_SKIP_FRAMES:<20}")
# print(f"{'Cache Hit Rate':<25} {cache_hit_rate:<20.1f}%")
# print(f"{'Total Frames Processed': <25} {frame_count:<20}")
# print(f"{'Cache Hits':<25} {cache_hits:<20}")
# print(f"{'Cache Misses':<25} {cache_misses:<20}")
#
# print("\n" + "=" * 60)
# print("MARKDOWN TABLE FORMAT:")
# print("=" * 60)
# print("\n| Model         | Device | Avg FPS | YOLO Skip | Embed Skip | Cache Hit Rate (%) |")
# print("|---------------|--------|---------|-----------|------------|--------------------|")
# print(f"| MobileFaceNet | GPU    | {avg_fps:.1f}    | {YOLO_SKIP_FRAMES:<9} | {EMBEDDING_SKIP_FRAMES:<10} | {cache_hit_rate:.1f}%               |")
# print("\n")

# BATAS DENGAN YANG TANPA OBS + LOAD LAMA

#!/usr/bin/env python3
"""
MOBILEFACENET - REALTIME WEBCAM TEST (GPU MODE ONLY)
---------------------------------------------------
Detektor    : YOLOv11n (GPU)
Vektorisasi : MobileFaceNet (GPU)
Fitur       : Auto-Blur Unknown, Real-time FPS, Face Tracking, Embedding Cache
Tambahan    : OBS Virtual Camera
Whitelist   : Mode A (Folder per Identitas)
"""

import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from mfnet import MobileFacenet
import pyvirtualcam

# ============================================================
# CONFIG
# ============================================================
THRESHOLD = 0.45
INPUT_RES = (640, 480)
WHITELIST_DIR = "../whitelist/"
PATH_YOLO_MODEL = "../model/model.pt"
PATH_MFNET_CKPT = "../model/068.ckpt"
COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)

YOLO_SKIP_FRAMES = 2
EMBEDDING_SKIP_FRAMES = 20
TRACKING_DISTANCE_THRESHOLD = 60
CACHE_MAX_AGE = 8

fps_history = []
max_fps_samples = 300

# ============================================================
# GPU SETUP
# ============================================================
device = torch.device("cuda")
print(f"[*] MODE: GPU ONLY")
print(f"[*] Device: {device}")
print(f"[*] CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("[!] ERROR: CUDA not available!")
    exit(1)

print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

CAM_INDEX = int(input("Pilih kamera (0,1,2,... ): ") or 0)

# ============================================================
# LOAD MODELS
# ============================================================
print("Loading YOLO model...")
yolo_model = YOLO(PATH_YOLO_MODEL).to(device)
print("✓ YOLO loaded on GPU")

print("Loading MobileFaceNet model...")
mfnet_model = MobileFacenet()

if os.path.exists(PATH_MFNET_CKPT):
    ckpt = torch.load(PATH_MFNET_CKPT, map_location="cpu")
    mfnet_model.load_state_dict(
        ckpt["net_state_dict"] if "net_state_dict" in ckpt else ckpt,
        strict=False
    )
    print("✓ MobileFaceNet checkpoint loaded")
else:
    print(f"ERROR: Checkpoint not found: {PATH_MFNET_CKPT}")
    exit(1)

mfnet_model.to(device).eval()

# ============================================================
# EMBEDDING CACHE
# ============================================================
embedding_cache = {}

def get_box_key(box):
    return "_".join(str(int(v / 10) * 10) for v in box)

def get_cached_embedding(box, now):
    key = get_box_key(box)
    if key in embedding_cache:
        if now - embedding_cache[key]["timestamp"] < CACHE_MAX_AGE:
            return embedding_cache[key]["embedding"]
        del embedding_cache[key]
    return None

def cache_embedding(box, emb, now):
    embedding_cache[get_box_key(box)] = {
        "embedding": emb,
        "timestamp": now
    }

def clean_old_cache(now):
    for k in list(embedding_cache.keys()):
        if now - embedding_cache[k]["timestamp"] > CACHE_MAX_AGE:
            del embedding_cache[k]

# ============================================================
# FACE UTILS
# ============================================================
@torch.no_grad()
def get_embedding(img):
    if img is None or img.size == 0:
        return None
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    emb = mfnet_model(tensor)
    emb = emb[0] if isinstance(emb, tuple) else emb
    emb = emb.flatten().cpu().numpy()
    return emb / np.linalg.norm(emb)

def get_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def blur_face(img):
    return cv2.GaussianBlur(img, (51, 51), 30)

def calculate_box_distance(a, b):
    ax, ay = (a[0]+a[2])/2, (a[1]+a[3])/2
    bx, by = (b[0]+b[2])/2, (b[1]+b[3])/2
    return ((ax-bx)**2 + (ay-by)**2) ** 0.5

def match_faces_to_previous(detections, prev_faces):
    matched, used = [], set()
    for pf in prev_faces:
        best_i, best_d = -1, 1e9
        for i, d in enumerate(detections):
            if i in used:
                continue
            dist = calculate_box_distance(pf["box"], d)
            if dist < TRACKING_DISTANCE_THRESHOLD and dist < best_d:
                best_i, best_d = i, dist
        if best_i != -1:
            nf = pf.copy()
            nf["box"] = detections[best_i]
            matched.append(nf)
            used.add(best_i)
    for i, d in enumerate(detections):
        if i not in used:
            matched.append({
                "box": d,
                "status": False,
                "score": 1.0
            })
    return matched

# ============================================================
# LOAD WHITELIST (MODE A)
# ============================================================
print("\nLoading whitelist...")
target_embeddings = []

for root, _, files in os.walk(WHITELIST_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(root, f))
            if img is None:
                continue
            results = yolo_model(img, verbose=False)
            for r in results:
                if len(r.boxes) > 0:
                    x1,y1,x2,y2 = map(int, r.boxes[0].xyxy[0])
                    emb = get_embedding(img[y1:y2, x1:x2])
                    if emb is not None:
                        target_embeddings.append(emb)
                        print(f"  ✓ {f}")
                    break

target_embeddings = np.array(target_embeddings)
print(f"Total: {len(target_embeddings)} embeddings loaded\n")

# ============================================================
# MAIN LOOP + OBS
# ============================================================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RES[1])

vcam = pyvirtualcam.Camera(
    width=INPUT_RES[0],
    height=INPUT_RES[1],
    fps=30
)
print(f"[✓] OBS Virtual Camera Active: {vcam.device}")

prev_time = time.time()
frame_count = 0
cache_hits = cache_misses = 0
current_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, INPUT_RES)
    display = frame.copy()
    now = time.time()

    fps = 1 / (now - prev_time) if now > prev_time else 0
    prev_time = now

    fps_history.append(fps)
    if len(fps_history) > max_fps_samples:
        fps_history.pop(0)

    frame_count += 1
    if frame_count % 100 == 0:
        clean_old_cache(now)

    if frame_count % YOLO_SKIP_FRAMES == 0 or frame_count == 1:
        detections = []
        for r in yolo_model(frame, stream=True, verbose=False):
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                if (x2-x1) > 10 and (y2-y1) > 10:
                    detections.append([x1,y1,x2,y2])
        current_faces = match_faces_to_previous(detections, current_faces)

    if frame_count % EMBEDDING_SKIP_FRAMES == 0 or frame_count == 1:
        for face in current_faces:
            x1,y1,x2,y2 = face["box"]
            crop = frame[y1:y2, x1:x2]
            emb = get_cached_embedding(face["box"], now)
            if emb is not None:
                cache_hits += 1
            else:
                cache_misses += 1
                emb = get_embedding(crop)
                if emb is not None:
                    cache_embedding(face["box"], emb, now)
            is_known = False
            score = 1.0
            if emb is not None and len(target_embeddings) > 0:
                dists = [get_distance(t, emb) for t in target_embeddings]
                score = min(dists)
                if score <= THRESHOLD:
                    is_known = True
            face["status"] = is_known
            face["score"] = score

    for face in current_faces:
        x1,y1,x2,y2 = face["box"]
        if not face["status"]:
            display[y1:y2, x1:x2] = blur_face(display[y1:y2, x1:x2])
            color = COLOR_UNKNOWN
            label = f"Unknown ({face['score']:.2f})"
        else:
            color = COLOR_KNOWN
            label = f"Me ({face['score']:.2f})"
        cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
        cv2.putText(display, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("MobileFaceNet - GPU Mode", display)

    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    vcam.send(rgb)
    vcam.sleep_until_next_frame()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# BATAS TIPE TAMBAHAN FITUR TIPE 3

# #!/usr/bin/env python3
# """
# MOBILEFACENET - REALTIME WEBCAM TEST (GPU MODE ONLY)
# ---------------------------------------------------
# Detektor    : YOLOv11n (GPU)
# Vektorisasi : MobileFaceNet (GPU)
# Fitur       : Auto-Blur Unknown, Real-time FPS, Face Tracking, Embedding Cache
# """
#
# import os
# import time
# import cv2
# import numpy as np
# import torch
# import pyvirtualcam
# from ultralytics import YOLO
# from mfnet import MobileFacenet
#
# # ============================================================
# # CONFIG
# # ============================================================
# THRESHOLD = 0.45
# INPUT_RES = (640, 480)
# WHITELIST_DIR = "../whitelist/"
# PATH_YOLO_MODEL = "../model/model.pt"
# PATH_MFNET_CKPT = "../model/068.ckpt"
#
# COLOR_KNOWN = (0, 255, 0)
# COLOR_UNKNOWN = (0, 0, 255)
#
# YOLO_SKIP_FRAMES = 2
# EMBEDDING_SKIP_FRAMES = 20
# TRACKING_DISTANCE_THRESHOLD = 60
# CACHE_MAX_AGE = 8
#
# FACE_TTL_FRAMES = 15
# STATUS_HOLD_FRAMES = 20
#
# fps_history = []
# max_fps_samples = 300
#
# # ============================================================
# # SETUP GPU
# # ============================================================
# device = torch.device("cuda")
# print(f"[*] MODE: GPU ONLY")
# print(f"[*] Device: {device}")
# print(f"[*] CUDA Available: {torch.cuda.is_available()}")
#
# if not torch.cuda.is_available():
#     print("[!] ERROR: CUDA not available!")
#     exit(1)
#
# print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
#
# CAM_INDEX = int(input("Pilih kamera (0,1,2,...): ") or 0)
#
# # ============================================================
# # LOAD MODELS
# # ============================================================
# print("Loading models...")
# yolo_model = YOLO(PATH_YOLO_MODEL).to(device)
#
# mfnet_model = MobileFacenet()
# ckpt = torch.load(PATH_MFNET_CKPT, map_location="cpu")
# if "net_state_dict" in ckpt:
#     mfnet_model.load_state_dict(ckpt["net_state_dict"], strict=False)
# else:
#     mfnet_model.load_state_dict(ckpt, strict=False)
#
# mfnet_model.to(device).eval()
# print("✓ Models loaded on GPU")
#
# # ============================================================
# # EMBEDDING CACHE
# # ============================================================
# embedding_cache = {}
#
# def get_box_key(box):
#     return "_".join(str(int(v / 10) * 10) for v in box)
#
# def get_cached_embedding(box, now):
#     k = get_box_key(box)
#     if k in embedding_cache:
#         if now - embedding_cache[k]["timestamp"] < CACHE_MAX_AGE:
#             return embedding_cache[k]["embedding"]
#         del embedding_cache[k]
#     return None
#
# def cache_embedding(box, emb, now):
#     embedding_cache[get_box_key(box)] = {
#         "embedding": emb,
#         "timestamp": now
#     }
#
# def clean_old_cache(now):
#     for k in list(embedding_cache.keys()):
#         if now - embedding_cache[k]["timestamp"] > CACHE_MAX_AGE:
#             del embedding_cache[k]
#
# # ============================================================
# # FACE UTILS
# # ============================================================
# @torch.no_grad()
# def get_embedding(img):
#     if img is None or img.size == 0:
#         return None
#     try:
#         img = cv2.resize(img, (112, 112))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = (img.astype(np.float32) - 127.5) / 128.0
#         img = np.transpose(img, (2, 0, 1))
#         t = torch.from_numpy(img).unsqueeze(0).to(device)
#         emb = mfnet_model(t)
#         if isinstance(emb, tuple):
#             emb = emb[0]
#         emb = emb.flatten().cpu().numpy()
#         return emb / np.linalg.norm(emb)
#     except:
#         return None
#
# def get_distance(a, b):
#     return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
# def blur_face(img):
#     return cv2.GaussianBlur(img, (51, 51), 30)
#
# def box_dist(a, b):
#     ax, ay = (a[0]+a[2])/2, (a[1]+a[3])/2
#     bx, by = (b[0]+b[2])/2, (b[1]+b[3])/2
#     return ((ax-bx)**2 + (ay-by)**2) ** 0.5
#
# def match_faces(new_boxes, faces, frame_id):
#     used = set()
#     updated = []
#
#     for f in faces:
#         best_i, best_d = -1, 1e9
#         for i, b in enumerate(new_boxes):
#             if i in used:
#                 continue
#             d = box_dist(f["box"], b)
#             if d < TRACKING_DISTANCE_THRESHOLD and d < best_d:
#                 best_i, best_d = i, d
#
#         if best_i != -1:
#             f["box"] = new_boxes[best_i]
#             f["ttl"] = FACE_TTL_FRAMES
#             f["last_seen"] = frame_id
#             used.add(best_i)
#             updated.append(f)
#         else:
#             f["ttl"] -= 1
#             if f["ttl"] > 0:
#                 updated.append(f)
#
#     for i, b in enumerate(new_boxes):
#         if i not in used:
#             updated.append({
#                 "box": b,
#                 "status": False,
#                 "score": 1.0,
#                 "ttl": FACE_TTL_FRAMES,
#                 "last_seen": frame_id,
#                 "hold": 0
#             })
#
#     return updated
#
# # ============================================================
# # LOAD WHITELIST
# # ============================================================
# print("\nLoading whitelist...")
# target_embeddings = []
#
# for root, _, files in os.walk(WHITELIST_DIR):
#     for f in files:
#         if f.lower().endswith(('.jpg', '.jpeg', '.png')):
#             img = cv2.imread(os.path.join(root, f))
#             if img is None:
#                 continue
#             results = yolo_model(img, verbose=False)
#             for r in results:
#                 if len(r.boxes) > 0:
#                     x1,y1,x2,y2 = map(int, r.boxes[0].xyxy[0])
#                     emb = get_embedding(img[y1:y2, x1:x2])
#                     if emb is not None:
#                         target_embeddings.append(emb)
#                         print(f"  ✓ {f}")
#                     break
#
# target_embeddings = np.array(target_embeddings)
# print(f"Total: {len(target_embeddings)} embeddings loaded\n")
#
# # ============================================================
# # MAIN LOOP
# # ============================================================
# cap = cv2.VideoCapture(CAM_INDEX)
# cap.set(3, INPUT_RES[0])
# cap.set(4, INPUT_RES[1])
#
# USE_VIRTUAL_CAM = True
# if USE_VIRTUAL_CAM:
#     vcam = pyvirtualcam.Camera(
#         width=INPUT_RES[0],
#         height=INPUT_RES[1],
#         fps=30
#     )
#     print(f"[✓] Virtual Camera aktif: {vcam.device}")
#
# prev_time = time.time()
# frame_count = 0
# cache_hits = cache_misses = 0
# current_faces = []
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame = cv2.resize(frame, INPUT_RES)
#     disp = frame.copy()
#     frame_count += 1
#     now = time.time()
#
#     fps = 1 / (now - prev_time) if now != prev_time else 0
#     prev_time = now
#
#     fps_history.append(fps)
#     if len(fps_history) > max_fps_samples:
#         fps_history.pop(0)
#
#     if frame_count % 100 == 0:
#         clean_old_cache(now)
#
#     if frame_count % YOLO_SKIP_FRAMES == 0 or frame_count == 1:
#         boxes = []
#         for r in yolo_model(frame, verbose=False):
#             for b in r.boxes:
#                 x1,y1,x2,y2 = map(int, b.xyxy[0])
#                 if x2-x1 > 10 and y2-y1 > 10:
#                     boxes.append([x1,y1,x2,y2])
#         current_faces = match_faces(boxes, current_faces, frame_count)
#
#     if frame_count % EMBEDDING_SKIP_FRAMES == 0 or frame_count == 1:
#         for f in current_faces:
#             x1,y1,x2,y2 = f["box"]
#             face = frame[y1:y2, x1:x2]
#
#             emb = get_cached_embedding(f["box"], now)
#             if emb is not None:
#                 cache_hits += 1
#             else:
#                 cache_misses += 1
#                 emb = get_embedding(face)
#                 if emb is not None:
#                     cache_embedding(f["box"], emb, now)
#
#             if emb is not None and len(target_embeddings):
#                 d = min(get_distance(t, emb) for t in target_embeddings)
#                 if d <= THRESHOLD:
#                     f["status"] = True
#                     f["hold"] = STATUS_HOLD_FRAMES
#                 else:
#                     if f["hold"] > 0:
#                         f["hold"] -= 1
#                     else:
#                         f["status"] = False
#                 f["score"] = d
#
#     for f in current_faces:
#         x1,y1,x2,y2 = f["box"]
#         if f["status"]:
#             color, label = COLOR_KNOWN, f"Me ({f['score']:.2f})"
#         else:
#             color, label = COLOR_UNKNOWN, f"Unknown ({f['score']:.2f})"
#             try:
#                 disp[y1:y2, x1:x2] = blur_face(disp[y1:y2, x1:x2])
#             except:
#                 pass
#
#         cv2.rectangle(disp,(x1,y1),(x2,y2),color,2)
#         cv2.putText(disp,label,(x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
#
#     cv2.imshow("MobileFaceNet - GPU Mode", disp)
#
#     if USE_VIRTUAL_CAM:
#         rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
#         vcam.send(rgb)
#         vcam.sleep_until_next_frame()
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# Note tambahan fitur tipe 3:
# - Face tidak langsung hilang (TTL)
# - Identitas ditahan sementara (HOLD)
# - Tracking antar frame berbasis bounding box
# - Embedding di-cache untuk efisiensi