from torch.utils.data import Dataset
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms

class RAVDESSDataset(Dataset):
    def __init__(self, video_paths, labels, fps=15):
        self.video_paths = video_paths
        self.labels = labels
        self.fps = fps
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = self._load_video_frames(self.video_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return frames, label

    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps_video // self.fps))
        frames = []
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if i % interval == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                frames.append(self.transform(img))
            i += 1
        cap.release()
        return torch.stack(frames)  # shape: [T, C, H, W]
