import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    """
    Extracts spatial features from video frames using a small custom CNN.
    Each frame is expected to be of shape [3, 224, 224].
    """

    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        """
        Forward pass for a single image or batch of images.
        Input: x of shape [B, 3, 224, 224]
        Output: shape [B, 128]
        """
        x = self.features(x)  # shape [B, 128, 1, 1]
        return x.view(x.size(0), -1)  # flatten to [B, 128]
