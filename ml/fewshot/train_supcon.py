import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam

from ml.encoder.encoder import Encoder
from ml.fewshot.loss import SupConLoss
from ml.utils.transforms import train_transform

def train_supcon(
    data_dir,
    epochs=10,
    batch_size=16,
    lr=1e-3,
    device="cpu"
):
    # Dataset and loader
    dataset = ImageFolder(data_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = Encoder().to(device)
    criterion = SupConLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            features = model(images)
            loss = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    return model
