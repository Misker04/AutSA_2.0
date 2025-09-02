
# AutSA - Autism Social Aid

AutSA is a real-time assistive AI system designed to support individuals with Autism Spectrum Disorder by recognizing emotional states from video input and simplifying spoken dialogue using speech recognition and synthesis.

---

## Features

- **Video-Based Emotion Recognition** using a custom CNN + Transformer model
- **Speech Recognition** via Whisper or Vosk
- **Rule-Based Dialogue Simplification** to enhance communication
- **Text-to-Speech (TTS)** feedback using pyttsx3 or gTTS
- Modular, Pythonic backend using Flask, OpenCV, PyTorch

---

## Installation Instructions

### Step 1: Clone the Repository

### Step 2: (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows
```

### Step 3: Install All Required Dependencies

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:
```
torch
torchvision
pandas
tqdm
opencv-python
scikit-learn
flask
pyttsx3
vosk
transformers     # If using Whisper ASR
```

---

## Dataset Setup

### Step 1: Download RAVDESS

- Download from: [RAVDESS on Zenodo](https://zenodo.org/record/1188976)
- Extract to `dataset/videos/`

```
dataset/
├── videos/
│   ├── 01-01-01-01-01-01-01.mp4
│   └── ...
└── labels.csv
```

### Step 2: Prepare `labels.csv`

```bash
python labels_generator.py
```

Ensure `labels.csv` has:
```csv
path,label
dataset/videos/Actor_01/03-01-01-01-01-01-01.mp4,3
dataset/videos/Actor_02/03-01-06-02-02-02-06.mp4,1
...
```

## Model Training

```bash
python training_loop.py
```

- Uses a custom CNN + Transformer model
- Includes `collate_video_batch` for handling variable-length video sequences

---

## Run the Full System

```bash
python app.py
```

### Workflow:

1. Record or upload a video
2. Extract audio and frames
3. Run ASR (Whisper or Vosk)
4. Simplify complex sentences
5. Recognize emotion from facial expressions
6. Generate appropriate, simplified speech response

---

## Acknowledgements

- RAVDESS Dataset by Livingstone & Russo
- Whisper by OpenAI

---
