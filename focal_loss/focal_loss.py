import torch
from torch import Tensor
from torch.nn import Module

class FocalLoss(Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Focal Loss for binary classification with sigmoid outputs.
        
        Args:
            alpha: Weighting factor for class imbalance (0-1)
            gamma: Focusing parameter (≥0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Tensor of shape (N, *) with values in [0,1] (after sigmoid)
            targets: Tensor of shape (N, *) with values in {0,1}
        """
        # Ensure inputs are in valid range
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        
        # Calculate binary cross entropy
        bce_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        
        # Calculate modulating factor
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

    def __repr__(self):
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction='{self.reduction}')"