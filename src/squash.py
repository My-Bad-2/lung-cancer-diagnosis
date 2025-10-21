import torch


def squash(vectors: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """The non-linear activation function used in Capsule Networks"""
    squared_norm: torch.Tensor = (vectors ** 2).sum(dim=dim, keepdim=True)
    scale: torch.Tensor = squared_norm / (1 + squared_norm)
    return scale * vectors / torch.sqrt(squared_norm + 1e-8)
