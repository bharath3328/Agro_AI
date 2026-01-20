import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from backend.ml.prototypes import compute_prototypes
from backend.ml.classifier import PrototypeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_path = r"backend\ml\encoder_supcon.pth"
    train_dir = r"data\fewshot\train"
    test_dir = r"data\fewshot\test"

    print(f"Loading prototypes from {encoder_path}...")
    prototypes, class_names = compute_prototypes(
        encoder_path, train_dir, device
    )

    classifier = PrototypeClassifier(
        encoder_path, prototypes, class_names, device
    )

    print(f"Loading test data from {test_dir}...")
    dataset = ImageFolder(test_dir)
    
    y_true = []
    y_pred = []

    print("Running inference on test set...")
    total = len(dataset)
    for i, (image_path, label) in enumerate(dataset.samples):
        if i % 10 == 0:
            print(f"Processing {i}/{total} images...")
        
        pred, conf = classifier.predict(image_path)
        true_label = class_names[label]
        
        y_true.append(true_label)
        y_pred.append(pred)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")

    # Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    plt.title("Confusion Matrix for Plant Disease Classification")
    plt.tight_layout()
    output_path = os.path.abspath("confusion_matrix.png")
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    evaluate_model()
