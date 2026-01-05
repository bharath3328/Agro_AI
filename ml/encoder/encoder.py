import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    """
    CNN Encoder for few-shot plant disease recognition
    Based on lightweight architecture (MobileNetV3)
    Inspired by Mu et al., 2024
    """

    def __init__(self, embedding_dim=128):
        super().__init__()

        # Load pretrained MobileNetV3
        backbone = models.mobilenet_v3_small(weights="DEFAULT")

        # Remove classification head
        self.feature_extractor = backbone.features

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection layer (embedding space)
        self.embedding = nn.Linear(576, embedding_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x
