"""
Loss functions for imbalanced binary classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FOCAL_GAMMA, FOCAL_ALPHA, LABEL_SMOOTH


class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss (Lin et al. 2017) + Label Smoothing.
    Handles residual class imbalance after balanced sampling.

    gamma : focusing parameter — higher → more focus on hard examples
    alpha : weight for the positive (oil) class
    smooth: label smoothing epsilon
    """
    def __init__(self,
                 gamma:  float = FOCAL_GAMMA,
                 alpha:  float = FOCAL_ALPHA,
                 smooth: float = LABEL_SMOOTH):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = alpha          # weight for class 1 (oil)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)

        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smooth / (n_cls - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth)

        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        # Cross-entropy with smooth labels
        ce = -(smooth_targets * log_probs).sum(dim=1)   # (N,)

        # Focal weight: (1 - p_t)^gamma
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(self.gamma)

        # Alpha weighting per sample
        alpha_t = torch.where(targets == 1,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1.0 - self.alpha))

        loss = (alpha_t * focal_weight * ce).mean()
        return loss


def mixup_data(x: torch.Tensor, y: torch.Tensor,
               alpha: float = 0.2, device: str = "cpu"):
    """Mixup augmentation on spectral vectors."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(x.size(0), device=device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
