import os
import csv

ravdess_dir = 'dataset/ravdess/videos'
label_csv = 'dataset/labels.csv'

def get_emotion_label(filename):
    code = int(filename.split('-')[2])
    return code - 1  # Convert 1–8 to 0–7

with open(label_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])

    for file in os.listdir(ravdess_dir):
        if file.endswith('.mp4'):
            full_path = os.path.join(ravdess_dir, file)
            label = get_emotion_label(file)
            writer.writerow([full_path, label])
