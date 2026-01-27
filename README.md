# Auto-Face-Blur-Debug

This repository is cloned from  
[Auto-Face-Blur](https://github.com/One-Of-Those-Organization/auto-faceblur)  
and is intended **for debugging purposes only**.

---

## Alur Kerja: Input → Proses → Output

- Input
  - Live camera: webcam/USB camera (device index), atau IP camera (RTSP/HTTP stream)
  - Opsional: folder whitelist berisi wajah yang tidak ingin diblur
- Proses
  - Tangkap frame dari sumber kamera secara real-time
  - Deteksi wajah dengan YOLOv11n (Ultralytics) dan OpenCV
  - Pencocokan wajah whitelist untuk mengecualikan blur
  - Terapkan pengaburan (Gaussian blur atau pixelation) pada wajah yang tidak di-whitelist
  - Render ke jendela preview dan/atau simpan ke berkas video
- Output
  - Live preview (jendela tampilan real-time)
  - Opsional: rekaman video hasil blur (mis. `camera_blurred_<timestamp>.mp4`)
  - Opsional: log ringkas jumlah wajah terdeteksi, diblur, dan di-whitelist

---

## Pre-requisites

Make sure the following folders exist **in the same directory** as this repository:

- `arcface`
- `dataset_uji` (folder with images to test – OPTIONAL)
- `facenet`
- `mobilefacenet`
- `model` (pretrained models: `068.ckpt` and `model.pt`)
- `vgg`
- `whitelist` (folder with whitelist images – **AT LEAST 1–2 FACE IMAGES**)
- `yolo_backbone`

> **NOTE**
>
> - You can obtain these folders by cloning the original repositories. 
> - **Python version must be 3.11.9**.  
>   Newer versions (e. g. 3.13+) are **not supported** due to incompatible libraries used in this project. 

---

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/MashuNakamura/auto-face-blur-debug.git
```

### 2. Navigate to the Project Directory

```bash
cd auto-face-blur-debug
```

### 3. Choose the Model to Debug

Select the embedding model you want to test (e.g. arcface, facenet, mobilefacenet, etc.).

Example (ArcFace):

```bash
cd arcface
```

### 4. Activate Python Virtual Environment

**Linux / macOS:**

```bash
source ./venv/bin/activate
```

**Windows:**

```bash
.\venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Script

```bash
python real-time-cpu.py  # CPU version
python real-time-gpu.py  # GPU version (if supported)
```

---

## Model-Specific Instructions

### ArcFace

#### Setup Environment

```bash
cd arcface
```

Activate virtual environment:

**Linux / macOS:**

```bash
source ./venv/bin/activate
```

**Windows:**

```bash
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu.py
```

---

### FaceNet

#### Setup Environment

```bash
cd facenet
source ./venv/bin/activate  # Linux / macOS
.\venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu.py
python real-time-gpu.py
```

---

### MobileFaceNet

#### Setup Environment

```bash
cd mobilefacenet
source ./venv/bin/activate  # Linux / macOS
.\venv\Scripts\activate     # Windows
pip install -r requirements. txt
```

#### Run Script

```bash
python real-time-cpu.py
python real-time-gpu.py
```

---

### VGG

#### Setup Environment

```bash
cd vgg
source ./venv/bin/activate  # Linux / macOS
.\venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu.py
```

---

### YOLO_Backbone

#### Setup Environment

```bash
cd yolo_backbone
source ./venv/bin/activate  # Linux / macOS
.\venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu. py
python real-time-gpu.py
```

---

## Notes

- **ArcFace** and **VGG** do not support GPU execution due to library compatibility issues.  
  Please use the CPU version for these models. 

- Ensure the **whitelist path** is correctly set when running the script.  
  If no whitelist is found, all detected faces will be blurred.

---

## License

This project follows the same license as the original [Auto-Face-Blur](https://github.com/One-Of-Those-Organization/auto-faceblur) repository
