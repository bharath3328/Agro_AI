import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        # Normalize feature vectors
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-comparisons
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        # Compute log probabilities safely
        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Handle batches with no positive pairs
        mask_sum = mask.sum(dim=1)
        valid = mask_sum > 0

        if valid.sum() == 0:
            # No valid contrastive pairs in batch
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask[valid] * log_prob[valid]).sum(dim=1) / mask_sum[valid]

        loss = -mean_log_prob_pos.mean()
        return loss
