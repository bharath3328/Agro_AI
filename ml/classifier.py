import torch
import torch.nn.functional as F
from PIL import Image

from ml.encoder import Encoder
from ml.transforms import inference_transform


class PrototypeClassifier:
    def __init__(self, encoder_path, prototypes, class_names, device="cpu"):
        self.device = device

        self.model = Encoder()
        self.model.load_state_dict(torch.load(encoder_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.prototypes = {
            k: v.to(device) for k, v in prototypes.items()
        }

        self.class_names = class_names

    def predict(self, image_path, threshold=0.6):

        image = Image.open(image_path).convert("RGB")
        image = inference_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image)
            embedding = F.normalize(embedding, dim=1)

        similarities = {}

        for label, prototype in self.prototypes.items():
            prototype = F.normalize(prototype.unsqueeze(0), dim=1)
            similarity = F.cosine_similarity(embedding, prototype)
            similarities[label] = similarity.item()

        best_label = max(similarities, key=similarities.get)
        best_score = similarities[best_label]

        print(f"DEBUG: Best Class: {self.class_names[best_label]} | Score: {best_score:.4f} | Threshold: {threshold:.4f}")

        if best_score < threshold:
            print(f"DEBUG: Rejected as UNKNOWN (Score {best_score:.4f} < {threshold:.4f})")
            return "UNKNOWN", float(best_score)

        sorted_scores = sorted(similarities.values(), reverse=True)
        if len(sorted_scores) > 1:
            margin = sorted_scores[0] - sorted_scores[1]
            if margin < 0.01: 
                print(f"DEBUG: Rejected as UNKNOWN (Ambiguous Margin {margin:.4f})")
                return "UNKNOWN", float(best_score)

        return self.class_names[best_label], float(best_score)
