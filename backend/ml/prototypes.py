import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from backend.ml.encoder import Encoder
from backend.ml.transforms import train_transform
import os

def compute_prototypes(encoder_path,data_dir,device="cpu"):
    model = Encoder()
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = ImageFolder(data_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    prototypes = {}
    counts = {}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            features = model(images)

            for feature, label in zip(features, labels):
                label = label.item()
                if label not in prototypes:
                    prototypes[label] = feature.clone()
                    counts[label] = 1
                else:
                    prototypes[label] += feature
                    counts[label] += 1

    for label in prototypes:
        prototypes[label] /= counts[label]

    return prototypes, dataset.classes
