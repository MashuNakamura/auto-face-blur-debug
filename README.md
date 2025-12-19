# Auto-Face-Blur-Debug

This repository is clone from [Auto-Face-Blur](https://github.com/One-Of-Those-Organization/auto-faceblur) for debugging purpose.

## Pre-requisites

Make sure you have this following folders in the same directory as this repository:

- arcface
- dataset_uji (folder with images to test - OPTIONAL)
- facenet
- mobilefacenet
- model (pretrained models 068.ckpt and model.pt)
- vgg
- whitelist (folder with whitelist images - AT LEAST 1-2 FACES)
- yolo_backbone

**NOTE** : You can get these folders from cloning this repositories, and make sure you have Python version 3.11.9 to make sure it work properly, because the newer like 3.13+ higher doesn't support some libraries used in this project.

## How to use

1. Clone this repository to your local machine.

```bash
git clone https://github.com/MashuNakamura/auto-face-blur-debug.git
```
2. Navigate to the project directory.
```bash
cd auto-face-blur-debug
```
3. Select which model embedding you want to debug (e.g., `arcface`, `mobilefacenet`, etc.). For example ArcFace you need to get into the `arcface` directory.
```bash
cd arcface
```
4. Activate your Python virtual environment.
```bash
source ./venv/bin/activate # Linux / macOS
.\venv\Scripts\activate    # Windows
```
5. Install the required dependencies.
```bash
pip install -r requirements.txt
```
6. Run the script (CPU or GPU version).
```bash
python real-time-cpu.py  # For CPU
python real-time-gpu.py  # For GPU
```

### Note

ArcFace and VGG doens't support GPU version due to some compatibility issues with the current libraries. Please use CPU version for these models.

## For Developers

Please run this and send me a result by record screen while using the model. For example you want to test ArcFace model, please run the `real-time-cpu.py` script in the `arcface` directory then send me the recorded video. Ensure the video have some :
1. Multiple faces (3 - 5 people)
2. Different angles (side face, looking up, looking down)
3. Different expressions (smile, frown, surprise)
4. While moving or walking
5. Different lighting conditions (bright light, low light) if possible

**NOTE** : Ensure you have whitelist path to your path where you run the script otherwise the script will blur all the face.

### ArcFace

#### Set Python Virtual Environment and Install Dependencies

```bash
cd arcface
source ./venv/bin/activate # Linux / macOS
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu.py  # For CPU
```

### FaceNet

#### Set Python Virtual Environment and Install Dependencies

```bash
cd facenet
source ./venv/bin/activate # Linux / macOS
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu.py  # For CPU
python real-time-gpu.py  # For GPU
```

### MobileFaceNet

#### Set Python Virtual Environment and Install Dependencies

```bash
cd mobilefacenet
source ./venv/bin/activate # Linux / macOS
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu.py  # For CPU
python real-time-gpu.py  # For GPU
```

### VGG

#### Set Python Virtual Environment and Install Dependencies

```bash
cd vgg
source ./venv/bin/activate # Linux / macOS
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu.py  # For CPU
```

### YOLO_Backbone

#### Set Python Virtual Environment and Install Dependencies

```bash
cd yolo_backbone
source ./venv/bin/activate # Linux / macOS
.\venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

#### Run Script

```bash
python real-time-cpu.py  # For CPU
python real-time-gpu.py  # For GPU
```