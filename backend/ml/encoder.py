import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights="DEFAULT")
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(576, embedding_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x
