from flask import Flask, render_template, request
from models.emotion_model import EmotionRecognitionModel
from utils.simplifier import simplify_sentence
from utils.response_gen import generate_responses

import torch
import torchvision.transforms as transforms
import cv2
from gtts import gTTS
from PIL import Image
import os
import whisper

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ASR model (Whisper)
asr_model = whisper.load_model("base")

# Load Emotion Model
model = EmotionRecognitionModel()
model.load_state_dict(torch.load("models/emotion_model.pth", map_location="cpu"))
model.eval()

# Emotion classes (example mapping)
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

# Frame preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_frames(video_path, fps=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(original_fps // fps))
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(transform(pil_img))
        i += 1
    cap.release()
    return frames

def predict_emotion(video_path):
    frames = extract_frames(video_path)
    with torch.no_grad():
        output = model(frames)
        pred_idx = torch.argmax(output).item()
    return EMOTIONS[pred_idx]

def transcribe_audio(video_path):
    result = asr_model.transcribe(video_path)
    return result["text"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_file = request.files['video']
        filename = video_file.filename
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(video_path)

        # Transcribe audio
        transcript = transcribe_audio(video_path)
        simplified = simplify_sentence(transcript)

        # Emotion recognition
        emotion = predict_emotion(video_path)

        # Response generation + TTS
        suggestions = generate_responses(emotion, simplified)
        tts = gTTS(simplified)
        tts_path = os.path.join("static", "response.mp3")
        tts.save(tts_path)

        return render_template("index.html",
                               original=transcript,
                               simplified=simplified,
                               emotion=emotion,
                               responses=suggestions,
                               tts_path=tts_path)

    return render_template("index.html")
