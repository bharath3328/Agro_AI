import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ml.prototypes import compute_prototypes
from ml.classifier import PrototypeClassifier

def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_path = "ml/encoder_supcon.pth"
    train_dir = "data/fewshot/train"
    test_dir = "data/fewshot/test"

    prototypes, class_names = compute_prototypes(
        encoder_path, train_dir, device
    )

    classifier = PrototypeClassifier(
        encoder_path, prototypes, class_names, device
    )

    dataset = ImageFolder(test_dir)
    correct = 0

    for image_path, label in dataset.samples:
        pred, conf = classifier.predict(image_path)
        true_label = class_names[label]

        if pred == true_label:
            correct += 1

    accuracy = (correct / len(dataset)) * 100
    print(f"Few-Shot Classification Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
