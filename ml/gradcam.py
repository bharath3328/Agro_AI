import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM implementation
    Based on Selvaraju et al., 2017
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        """Standard Grad-CAM for classification heads (deprecated for this project)"""
        self.model.zero_grad()
        output = self.model(input_tensor)

        score = output[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

    def generate_from_prototype(self, input_tensor, prototype):
        """
        Grad-CAM for Metric Learning (Prototype-based)
        Explains: "Which parts of the image make it similar to this prototype?"
        """
        self.model.zero_grad()
        
        # Forward pass to get embedding
        embedding = self.model(input_tensor)
        
        # Normalize for cosine similarity
        embedding_norm = F.normalize(embedding, dim=1)
        prototype_norm = F.normalize(prototype.unsqueeze(0), dim=1)
        
        # Score is the cosine similarity
        score = (embedding_norm * prototype_norm).sum()
        
        # Backward pass from similarity score
        score.backward()

        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam
