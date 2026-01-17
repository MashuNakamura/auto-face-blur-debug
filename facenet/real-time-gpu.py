# #!/usr/bin/env python3
# """
# FACENET - REALTIME WEBCAM TEST (GPU MODE ONLY)
# ---------------------------------------------------
# Detektor    : YOLOv11n (GPU)
# Vektorisasi : FaceNet (InceptionResnetV1 - VGGFace2)
# Fitur       : Auto-Blur Unknown, Real-time FPS, Face Tracking, Embedding Cache
# Tambahan    : (OPSIONAL) Virtual Camera OBS / Google Meet
# """
#
# import os
# import time
# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO
# from facenet_pytorch import InceptionResnetV1
# import pyvirtualcam
#
# # ============================================================
# # CONFIG
# # ============================================================
# THRESHOLD = 0.38
# INPUT_RES = (640, 480)
# WHITELIST_DIR = "../whitelist/"
# PATH_YOLO_MODEL = "../model/model.pt"
# COLOR_KNOWN = (0, 255, 0)
# COLOR_UNKNOWN = (0, 0, 255)
#
# YOLO_SKIP_FRAMES = 2
# EMBEDDING_SKIP_FRAMES = 30
# TRACKING_DISTANCE_THRESHOLD = 100
# CACHE_MAX_AGE = 15
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
# facenet_model = InceptionResnetV1(
#     pretrained='vggface2',
#     classify=False
# ).to(device).eval()
#
# print("✓ Models loaded on GPU")
#
# # ============================================================
# # EMBEDDING CACHE SYSTEM
# # ============================================================
# embedding_cache = {}
#
# def get_box_key(box):
#     return "_".join(str(int(v / 10) * 10) for v in box)
#
# def get_cached_embedding(box, now):
#     key = get_box_key(box)
#     if key in embedding_cache:
#         age = now - embedding_cache[key]['timestamp']
#         if age < CACHE_MAX_AGE:
#             return embedding_cache[key]['embedding']
#         del embedding_cache[key]
#     return None
#
# def cache_embedding(box, emb, now):
#     embedding_cache[get_box_key(box)] = {
#         'embedding': emb,
#         'timestamp': now
#     }
#
# def clean_old_cache(now):
#     for k in list(embedding_cache.keys()):
#         if now - embedding_cache[k]['timestamp'] > CACHE_MAX_AGE:
#             del embedding_cache[k]
#
# # ============================================================
# # FACE UTILS
# # ============================================================
# @torch.no_grad()
# def get_embedding(img):
#     if img is None or img.size == 0:
#         return None
#     img = cv2.resize(img, (160, 160))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     tensor = torch.from_numpy(img).float().to(device)
#     tensor = tensor / 255.0
#     tensor = tensor.permute(2, 0, 1).unsqueeze(0)
#     return facenet_model(tensor).flatten().cpu().numpy()
#
# def get_distance(a, b):
#     return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
# def blur_face(img):
#     return cv2.GaussianBlur(img, (51, 51), 30)
#
# def calculate_box_distance(a, b):
#     ax, ay = (a[0]+a[2])/2, (a[1]+a[3])/2
#     bx, by = (b[0]+b[2])/2, (b[1]+b[3])/2
#     return ((ax-bx)**2 + (ay-by)**2) ** 0.5
#
# def match_faces_to_previous(detections, prev_faces):
#     matched = []
#     used = set()
#
#     for pf in prev_faces:
#         best_i, best_d = -1, 1e9
#         for i, d in enumerate(detections):
#             if i in used:
#                 continue
#             dist = calculate_box_distance(pf['box'], d)
#             if dist < TRACKING_DISTANCE_THRESHOLD and dist < best_d:
#                 best_i, best_d = i, dist
#
#         if best_i != -1:
#             nf = pf.copy()
#             nf['box'] = detections[best_i]
#             matched.append(nf)
#             used.add(best_i)
#
#     for i, d in enumerate(detections):
#         if i not in used:
#             matched.append({
#                 'box': d,
#                 'status': False,
#                 'score': 1.0,
#                 'needs_recognition': True
#             })
#
#     return matched
#
# # ============================================================
# # LOAD WHITELIST (TIPE 2)
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
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RES[0])
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RES[1])
#
# # ============================================================
# # INIT OBS VIRTUAL CAMERA
# # ============================================================
# vcam = pyvirtualcam.Camera(
#     width=INPUT_RES[0],
#     height=INPUT_RES[1],
#     fps=30
# )
# print(f"[✓] OBS Virtual Camera Active: {vcam.device}")
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
#     display = frame.copy()
#
#     now = time.time()
#     fps = 1 / (now - prev_time) if now > prev_time else 0
#     prev_time = now
#
#     fps_history.append(fps)
#     if len(fps_history) > max_fps_samples:
#         fps_history.pop(0)
#
#     frame_count += 1
#     if frame_count % 100 == 0:
#         clean_old_cache(now)
#
#     if frame_count % YOLO_SKIP_FRAMES == 0 or frame_count == 1:
#         detections = []
#         for r in yolo_model(frame, stream=True, verbose=False):
#             for b in r.boxes:
#                 x1,y1,x2,y2 = map(int, b.xyxy[0])
#                 if (x2-x1) > 10 and (y2-y1) > 10:
#                     detections.append([x1,y1,x2,y2])
#         current_faces = match_faces_to_previous(detections, current_faces)
#
#     if frame_count % EMBEDDING_SKIP_FRAMES == 0 or frame_count == 1:
#         for face in current_faces:
#             x1,y1,x2,y2 = face['box']
#             crop = frame[y1:y2, x1:x2]
#             is_known = False
#             score = 1.0
#
#             emb = get_cached_embedding(face['box'], now)
#             if emb is not None:
#                 cache_hits += 1
#             else:
#                 cache_misses += 1
#                 emb = get_embedding(crop)
#                 if emb is not None:
#                     cache_embedding(face['box'], emb, now)
#
#             if emb is not None and len(target_embeddings) > 0:
#                 dists = [get_distance(t, emb) for t in target_embeddings]
#                 score = min(dists)
#                 if score <= THRESHOLD:
#                     is_known = True
#
#             face['status'] = is_known
#             face['score'] = score
#
#     for face in current_faces:
#         x1,y1,x2,y2 = face['box']
#         if not face['status']:
#             display[y1:y2, x1:x2] = blur_face(display[y1:y2, x1:x2])
#             color = COLOR_UNKNOWN
#             label = f"Unknown ({face['score']:.2f})"
#         else:
#             color = COLOR_KNOWN
#             label = f"Me ({face['score']:.2f})"
#
#         cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
#         cv2.putText(display, label, (x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#     cv2.imshow("FaceNet - GPU Mode", display)
#
#     # Send to OBS Virtual Camera
#     rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
#     vcam.send(rgb)
#     vcam.sleep_until_next_frame()
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# Kode Lama diatas
# Kode baru dengan fitur auto-enroll whitelist menggunakan webcam dan GPU dibawah

#!/usr/bin/env python3
"""
FACENET - REALTIME WEBCAM TEST (GPU MODE ONLY)
---------------------------------------------------
Detektor    : YOLOv11n (GPU)
Vektorisasi : FaceNet (InceptionResnetV1 - VGGFace2)
Fitur       : Auto-Blur Unknown, Real-time FPS, Face Tracking, Embedding Cache
"""

import os
import time
import cv2
import numpy as np
import torch
import pyvirtualcam
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# ============================================================
# CONFIG
# ============================================================
THRESHOLD = 0.38
INPUT_RES = (640, 480)
WHITELIST_DIR = "../whitelist/"
PATH_YOLO_MODEL = "../model/model.pt"
COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)

YOLO_SKIP_FRAMES = 2
EMBEDDING_SKIP_FRAMES = 30
TRACKING_DISTANCE_THRESHOLD = 100
CACHE_MAX_AGE = 15

FACE_TTL_FRAMES = 15
STATUS_HOLD_FRAMES = 20

fps_history = []
max_fps_samples = 300

# ============================================================
# SETUP
# ============================================================
device = torch.device("cuda")
print(f"[*] MODE: GPU ONLY")
print(f"[*] Device: {device}")
print(f"[*] CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[*] GPU:  {torch.cuda.get_device_name(0)}")
else:
    print("[!] ERROR: CUDA not available!")
    exit(1)

CAM_INDEX = int(input("Pilih kamera (0,1,2,...): ") or 0)

print("Loading models...")
yolo_model = YOLO(PATH_YOLO_MODEL).to(device)
facenet_model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device).eval()
print("✓ Models loaded on GPU")

# ============================================================
# EMBEDDING CACHE
# ============================================================
embedding_cache = {}

def get_box_key(box):
    return "_".join(str(int(v / 10) * 10) for v in box)

def get_cached_embedding(box, now):
    k = get_box_key(box)
    if k in embedding_cache:
        if now - embedding_cache[k]['timestamp'] < CACHE_MAX_AGE:
            return embedding_cache[k]['embedding']
        del embedding_cache[k]
    return None

def cache_embedding(box, emb, now):
    embedding_cache[get_box_key(box)] = {'embedding': emb, 'timestamp': now}

def clean_old_cache(now):
    for k in list(embedding_cache.keys()):
        if now - embedding_cache[k]['timestamp'] > CACHE_MAX_AGE:
            del embedding_cache[k]

# ============================================================
# UTILS
# ============================================================
@torch.no_grad()
def get_embedding(img):
    if img is None or img.size == 0:
        return None
    try:
        img = cv2.resize(img, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).float().to(device) / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        return facenet_model(t).flatten().cpu().numpy()
    except:
        return None

def get_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def blur_face(img):
    return cv2.GaussianBlur(img, (51, 51), 30)

def box_dist(b1, b2):
    c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
    c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

def match_faces(new_boxes, faces, frame_id):
    used = set()
    updated = []

    for f in faces:
        best_i, best_d = -1, 1e9
        for i, b in enumerate(new_boxes):
            if i in used: continue
            d = box_dist(f['box'], b)
            if d < TRACKING_DISTANCE_THRESHOLD and d < best_d:
                best_d, best_i = d, i

        if best_i != -1:
            f['box'] = new_boxes[best_i]
            f['ttl'] = FACE_TTL_FRAMES
            f['last_seen'] = frame_id
            used.add(best_i)
            updated.append(f)
        else:
            f['ttl'] -= 1
            if f['ttl'] > 0:
                updated.append(f)

    for i, b in enumerate(new_boxes):
        if i not in used:
            updated.append({
                'box': b,
                'status': False,
                'score': 1.0,
                # 'needs_recognition': True,
                'ttl': FACE_TTL_FRAMES,
                'last_seen': frame_id,
                'hold': 0
            })
    return updated

# # ============================================================
# # LOAD WHITELIST
# # ============================================================
# print("\nLoading whitelist...")
# targets = []
#
# os.makedirs(WHITELIST_DIR, exist_ok=True)
# for f in os.listdir(WHITELIST_DIR):
#     if f.lower().endswith(('.jpg','.png','.jpeg')):
#         img = cv2.imread(os.path.join(WHITELIST_DIR, f))
#         r = yolo_model(img, verbose=False)[0]
#         if len(r.boxes):
#             x1,y1,x2,y2 = map(int, r.boxes[0].xyxy[0])
#             emb = get_embedding(img[y1:y2, x1:x2])
#             if emb is not None:
#                 targets.append(emb)
#                 print(f"  ✓ {f}")
#
# targets = np.array(targets)
# print(f"Total: {len(targets)} embeddings\n")

# Pembeda Tipe 1 dan Tipe 2
print("\nLoading whitelist...")
target_embeddings = []

for root, _, files in os.walk(WHITELIST_DIR):
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, fname)
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
                        print(f"  ✓ {path}")
                    break

target_embeddings = np.array(target_embeddings)
print(f"Total: {len(target_embeddings)} embeddings loaded\n")

# ============================================================
# MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(CAM_INDEX)

# ============================================================
# VIRTUAL CAMERA SETUP
# ============================================================
USE_VIRTUAL_CAM = True  # toggle aman

if USE_VIRTUAL_CAM:
    vcam = pyvirtualcam.Camera(
        width=INPUT_RES[0],
        height=INPUT_RES[1],
        fps=30,
        print_fps=False
    )
    print(f"[✓] Virtual Camera aktif: {vcam.device}")

cap.set(3, INPUT_RES[0])
cap.set(4, INPUT_RES[1])

prev_time = time.time()
frame_count = 0
cache_hits = cache_misses = 0
current_faces = []

print("="*50)
print("FACENET - GPU MODE")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, INPUT_RES)
    disp = frame.copy()
    frame_count += 1
    now = time.time()

    fps = 1 / (now - prev_time) if now != prev_time else 0
    prev_time = now
    fps_history.append(fps)
    if len(fps_history) > max_fps_samples:
        fps_history.pop(0)

    if frame_count % 100 == 0:
        clean_old_cache(now)

    yolo_status = "CACHED"
    if frame_count % YOLO_SKIP_FRAMES == 0 or frame_count == 1:
        yolo_status = "DETECTING"
        boxes = []
        for r in yolo_model(frame, verbose=False):
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                if x2-x1 > 10 and y2-y1 > 10:
                    boxes.append([x1,y1,x2,y2])
        current_faces = match_faces(boxes, current_faces, frame_count)

    embed_status = "CACHED"
    if frame_count % EMBEDDING_SKIP_FRAMES == 0 or frame_count == 1:
        embed_status = "RECOGNIZING"
        for f in current_faces:
            x1,y1,x2,y2 = f['box']
            face = frame[y1:y2, x1:x2]
            emb = get_cached_embedding(f['box'], now)
            if emb is not None:
                cache_hits += 1
            else:
                cache_misses += 1
                emb = get_embedding(face)
                if emb is not None:
                    cache_embedding(f['box'], emb, now)

            if emb is not None and len(target_embeddings):
                d = min(get_distance(t, emb) for t in target_embeddings)
                if d <= THRESHOLD:
                    f['status'] = True
                    f['hold'] = STATUS_HOLD_FRAMES
                else:
                    if f['hold'] > 0:
                        f['hold'] -= 1
                    else:
                        f['status'] = False
                f['score'] = d

    for f in current_faces:
        x1,y1,x2,y2 = f['box']
        if f['status']:
            color, label = COLOR_KNOWN, f"Me ({f['score']:.2f})"
        else:
            color, label = COLOR_UNKNOWN, f"Unknown ({f['score']:.2f})"
            try:
                disp[y1:y2,x1:x2] = blur_face(disp[y1:y2,x1:x2])
            except: pass

        cv2.rectangle(disp,(x1,y1),(x2,y2),color,2)
        cv2.putText(disp,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    avg_fps = np.mean(fps_history)
    cache_rate = cache_hits / max(1, cache_hits + cache_misses)

    cv2.rectangle(disp,(0,0),(640,55),(0,0,0),-1)
    cv2.putText(disp,f"[GPU] FaceNet | FPS: {fps:.1f} (Avg: {avg_fps:.1f}) | Cache: {len(embedding_cache)}",
                (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
    cv2.putText(disp,f"YOLO: {yolo_status} | EMBED: {embed_status} | Hit Rate: {cache_rate:.1%}",
                (10,40),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)

    cv2.imshow("FaceNet - GPU Mode", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if USE_VIRTUAL_CAM:
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        vcam.send(rgb)
        vcam.sleep_until_next_frame()

cap.release()
cv2.destroyAllWindows()

# Note tambahan fitur tipe 3:
# Face tidak langsung hilang -> ditambah memori wajah antar frame (TTL)
# Identitas ditahan sementara -> ditambah status hold
# Wajah yang sama dilacak antar frame -> face dicocokkan berdasarkan posisi bounding box
# Embedding tidak dihitung ulang terus