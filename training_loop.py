import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, random_split
from models.emotion_model import EmotionRecognitionModel
from dataset_loader import RAVDESSDataset
from tqdm import tqdm
import os

# ✅ Custom collate_fn for batching variable-length video sequences
def collate_video_batch(batch):
    frames_batch, labels = zip(*batch)  # list of [T, C, H, W], labels
    max_len = max([f.shape[0] for f in frames_batch])

    padded_batch = []
    for f in frames_batch:
        pad_len = max_len - f.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, *f.shape[1:]))  # [pad_len, C, H, W]
            f = torch.cat([f, pad], dim=0)
        padded_batch.append(f)

    return torch.stack(padded_batch), torch.tensor(labels)

# 📥 Load dataset paths + labels
df = pd.read_csv("dataset/labels.csv")
video_paths = df['path'].tolist()
labels = df['label'].tolist()

# ⚙️ Hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
VAL_SPLIT = 0.2
LEARNING_RATE = 0.001

# 🚀 Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionRecognitionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# 📊 Dataset and split
full_dataset = RAVDESSDataset(video_paths, labels)
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_video_batch)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_video_batch)

# 🧠 Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_preds, train_targets = [], []

    for batch_frames, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        for i in range(batch_frames.shape[0]):  # loop over batch
            frames = batch_frames[i].to(device)  # [T, C, H, W]
            label = batch_labels[i].to(device)

            output = model(frames)
            loss = criterion(output.unsqueeze(0), label.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.argmax(output).item())
            train_targets.append(label.item())

    train_acc = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, average='weighted')
    print(f"📊 Epoch {epoch+1} Training | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

    # 🧪 Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch_frames, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            for i in range(batch_frames.shape[0]):
                frames = batch_frames[i].to(device)
                label = batch_labels[i].to(device)

                output = model(frames)
                val_preds.append(torch.argmax(output).item())
                val_targets.append(label.item())

    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average='weighted')
    print(f"✅ Epoch {epoch+1} Validation | Acc: {val_acc * 100:.2f}% | F1: {val_f1:.4f}")

# 💾 Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/emotion_model.pth")
print("✅ Final model saved to models/emotion_model.pth")
