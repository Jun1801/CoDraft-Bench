import torch
import torch.nn as nn
import torch.nn.functional as F

class RankAwareFocalLoss(nn.Module):
    def __init__(self, num_classes=5, gamma=2.0, alpha=0.5):
        
        super(RankAwareFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('rank_values', torch.arange(num_classes).float())

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        focal_loss = focal_loss.mean()
        probs = F.softmax(logits, dim=-1)

        rank_values = self.rank_values.to(logits.device)
        expected_ranks = torch.sum(probs * rank_values, dim=-1)

        rank_loss = F.mse_loss(expected_ranks, targets.float())

        total_loss = focal_loss + (self.alpha * rank_loss)

        return total_loss