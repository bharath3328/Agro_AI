import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from backend.ml.prototypes import compute_prototypes

def plot_similarity_heatmap():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_path = r"backend\ml\encoder_supcon.pth"
    train_dir = r"data\fewshot\train"
    
    print(f"Loading prototypes from {encoder_path}...")
    prototypes, class_names = compute_prototypes(
        encoder_path, train_dir, device
    )
    sorted_indices = np.argsort(class_names)
    start_names = [class_names[i] for i in sorted_indices]
    proto_tensor = torch.stack([prototypes[i] for i in sorted_indices])
    proto_tensor = F.normalize(proto_tensor, dim=1)
    similarity_matrix = torch.mm(proto_tensor, proto_tensor.t()).cpu().numpy()
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix, cmap='RdYlBu_r', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    tick_marks = np.arange(len(start_names))
    plt.xticks(tick_marks, start_names, rotation=90, fontsize=10)
    plt.yticks(tick_marks, start_names, fontsize=10)
    thresh = similarity_matrix.max() / 2
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            plt.text(j, i, format(similarity_matrix[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if similarity_matrix[i, j] > 0.5 else "black",
                     fontsize=8)
            
    plt.title("Fig. 1. Feature Similarity Heatmap (Prototype Embeddings)")
    plt.tight_layout()
    
    output_path = os.path.abspath("feature_similarity_heatmap.png")
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    plot_similarity_heatmap()
