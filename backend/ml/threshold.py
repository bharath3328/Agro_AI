import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from backend.ml.encoder import Encoder
from backend.ml.prototypes import compute_prototypes
from backend.ml.transforms import train_transform

def compute_open_set_threshold(encoder_path,train_dir,device="cpu",percentile=0.5):
    model = Encoder()
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    model.to(device)
    model.eval()

    prototypes, class_names = compute_prototypes(encoder_path, train_dir, device)
    dataset = ImageFolder(train_dir, transform=train_transform)
    similarities = []

    with torch.no_grad():
        for img, label in dataset:
            img = img.unsqueeze(0).to(device)
            emb = model(img)
            emb = F.normalize(emb, dim=1)

            proto = prototypes[label].unsqueeze(0).to(device)
            proto = F.normalize(proto, dim=1)

            sim = F.cosine_similarity(emb, proto)
            similarities.append(sim.item())

    threshold = torch.quantile(torch.tensor(similarities),percentile / 100).item()

    return threshold
