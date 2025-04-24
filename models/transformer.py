import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """
    Transformer encoder to model temporal dynamics of facial expressions over video frames.
    Input shape: [T, B, D] → Output: [B, num_emotions]
    """

    def __init__(self, input_dim=128, n_heads=4, n_layers=2, num_classes=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            activation='relu',
            batch_first=False  # Required for [T, B, D] input format
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        x: [T, B, D] - sequence of feature vectors from CNN
        Output: logits [B, num_classes]
        """
        encoded = self.encoder(x)           # [T, B, D]
        pooled = encoded.mean(dim=0)        # [B, D] (mean across time)
        logits = self.classifier(pooled)    # [B, num_classes]
        return logits.squeeze(0)            # Remove batch dim for single-sample prediction
