import torch
import torch.nn as nn


class CapsuleLoss(nn.Module):
    """The margin loss function for Capsule Networks."""

    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, votes: torch.Tensor, labels) -> torch.Tensor:
        # v shape: (batch, num_classes, 24)
        v_norm: torch.Tensor = torch.sqrt((votes ** 2).sum(dim=2, keepdim=True) + 1e-8).squeeze()

        # labels should be one-hot encoded
        left: torch.Tensor = torch.relu(0.9 - v_norm).pow(2)
        right: torch.Tensor = torch.relu(v_norm - 0.1).pow(2)

        margin_loss: torch.Tensor = (labels * left + 0.5 * (1.0 - labels) * right).sum()
        return margin_loss / votes.size(0)
   