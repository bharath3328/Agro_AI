import torch
import torch.nn.functional as F
from backend.ml.encoder import Encoder
from backend.ml.prototypes import compute_prototypes
from torchvision import transforms
from PIL import Image
import os

def test_openset():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_path = r"backend\ml\encoder_supcon.pth"
    train_dir = r"data\fewshot\train"

    from backend.ml.threshold import compute_open_set_threshold
    threshold = compute_open_set_threshold(encoder_path, train_dir, device=device)
    print(f"Computed Threshold: {threshold:.4f}")
    
    model = Encoder()
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    model.to(device)
    model.eval()
    prototypes, class_names = compute_prototypes(encoder_path, train_dir, device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nTesting on Known Image (Potato Early Blight)...")
    known_image_path = r"data\fewshot\test\Potato_Early_Blight\042135e2-e126-4900-9212-d42d900b8125___RS_Early.B 8791.JPG"
    img_known = Image.open(known_image_path).convert("RGB")
    data_known = transform(img_known).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb_known = model(data_known)
        emb_known = F.normalize(emb_known, dim=1)
        
        max_sim = -1
        for label, proto in prototypes.items():
            sim = F.cosine_similarity(emb_known, proto.unsqueeze(0).to(device)).item()
            if sim > max_sim:
                max_sim = sim
                
    print(f"Max Similarity: {max_sim:.4f}")
    if max_sim >= threshold:
        print("Prediction: Known Disease")
    else:
        print("Prediction: Unknown Disease (Open-set)")
    print("\nTesting on Simulated Unknown Image (Random Noise)...")
    img_unknown = torch.rand(1, 3, 224, 224).to(device) # Random noise
    img_unknown = (img_unknown - 0.5) / 0.5 
    
    with torch.no_grad():
        emb_unknown = model(img_unknown)
        emb_unknown = F.normalize(emb_unknown, dim=1)
        
        max_sim_unknown = -1
        for label, proto in prototypes.items():
            sim = F.cosine_similarity(emb_unknown, proto.unsqueeze(0).to(device)).item()
            if sim > max_sim_unknown:
                max_sim_unknown = sim
                
    print(f"Max Similarity: {max_sim_unknown:.4f}")
    if max_sim_unknown >= threshold:
        print("Prediction: Known Disease")
    else:
        print("Prediction: Unknown Disease (Open-set)")

if __name__ == "__main__":
    test_openset()
