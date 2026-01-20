import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from backend.config import settings
from backend.ml.encoder import Encoder
from backend.ml.transforms import train_transform
from backend.ml.prototypes import compute_prototypes
from backend.ml.threshold import compute_open_set_threshold
from backend.ml.loss import SupConLoss

def fine_tune():
    print("Starting Supervised Contrastive Fine-Tuning (SupCon)...")
    print("-" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dir = settings.TRAIN_DATA_DIR
    encoder_path = settings.ENCODER_PATH
    
    if not os.path.exists(train_dir):
        print("Data directory not found!")
        return

    dataset = ImageFolder(train_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    num_classes = len(dataset.classes)
    print(f"Found {num_classes} classes. Training on {len(dataset)} images.")

    print("Loading Encoder...")
    model = Encoder(embedding_dim=128)
    if os.path.exists(encoder_path):
        try:
            model.load_state_dict(torch.load(encoder_path, map_location=device))
            print("Loaded existing weights.")
        except:
            print("Could not load existing weights, starting from scratch (ImageNet/Random).")
    
    model.to(device)

    projection_head = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128)
    ).to(device)
    criterion = SupConLoss(temperature=0.07)
    optimizer = optim.Adam(
        list(model.parameters()) + list(projection_head.parameters()), 
        lr=0.0001
    )
    EPOCHS = 20
    model.train()
    projection_head.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)            
            optimizer.zero_grad()
            embeddings = model(images)
            projected = projection_head(embeddings)
            projected = F.normalize(projected, dim=1)
            loss = criterion(projected, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | SupCon Loss: {epoch_loss:.4f}")
    print("-" * 50)
    print("Saving fine-tuned encoder (Head discarded)...")
    torch.save(model.state_dict(), encoder_path)
    print(f"Encoder saved to {encoder_path}")
    print("\nUpdating Prototypes & Threshold (using clean Encoder output)...")
    model.eval() 
    prototypes, class_names = compute_prototypes(encoder_path, train_dir, device)
    threshold = compute_open_set_threshold(encoder_path, train_dir, device, percentile=10.0) 
    
    print(f"Re-computed prototypes for {len(class_names)} classes.")
    print(f"New Open-Set Threshold: {threshold:.4f}")
    
    print("\nModel successfully fine-tuned with SupCon!")
    print("Please restart your backend server.")

if __name__ == "__main__":
    fine_tune()
