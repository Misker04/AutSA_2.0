import torch
import torch.nn as nn
from .cnn import CNNFeatureExtractor
from .transformer import TransformerEncoder

class EmotionRecognitionModel(nn.Module):
    """
    Combines a CNN (spatial feature extractor) with a Transformer encoder (temporal sequence model)
    to recognize emotions from a sequence of video frames.
    """

    def __init__(self, feature_dim=128, num_emotions=8):
        super(EmotionRecognitionModel, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.transformer = TransformerEncoder(input_dim=feature_dim, num_classes=num_emotions)

    def forward(self, frame_seq):
        """
        frame_seq: list or tensor of shape [T, 3, 224, 224]
        Returns: logits of shape [num_emotions]
        """
        # Convert list of frames into a batch
        if isinstance(frame_seq, list):
            frame_seq = torch.stack(frame_seq)  # [T, 3, 224, 224]

        # Apply CNN to each frame independently
        features = []
        for frame in frame_seq:
            feat = self.cnn(frame.unsqueeze(0))  # [1, 128]
            features.append(feat.squeeze(0))     # [128]

        sequence_tensor = torch.stack(features)  # [T, 128]
        sequence_tensor = sequence_tensor.unsqueeze(1)  # [T, 1, 128] for Transformer

        logits = self.transformer(sequence_tensor)  # [num_emotions]
        return logits
