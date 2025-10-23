import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import vit_b_16
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

import pandas as pd
import os
import numpy as np
import json

def squash(s, dim=-1):
    squared_norm = torch.sum(torch.pow(s, 2), dim=dim, keepdim=True)
    norm = torch.sqrt(squared_norm)
    return (squared_norm / (1.0 + squared_norm)) * (s / norm)

class PrimaryCapsLayer(nn.Module):
    """
    Convert ViT tokens into Primary Capsules.
    """
    def __init__(self, vit_dim, num_capsules, capsule_dim):
        super(PrimaryCapsLayer, self).__init__()

        self.capsule_dim_total: int = num_capsules * capsule_dim
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        self.linear = nn.Sequential(
            nn.Linear(vit_dim, self.capsule_dim_total),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), -1, self.capsule_dim)
        return x


class SaliencyGuidedRouting(nn.Module):
    """
    Implements Dynamic Routing with Saliency-Guided Attention Modulation (SGAR).
    Attention weights modulate the initial routing logits b_ij.
    """
    def __init__(self, in_capsules, out_capsules, in_dim, out_dim, iterations=3):
        super(SaliencyGuidedRouting, self).__init__()

        self.in_capsules: int = in_capsules
        self.out_capsules: int = out_capsules
        self.iterations = iterations
        self.W = nn.Parameter(
            torch.randn(in_capsules, out_capsules, in_dim, out_dim),
        )

    def forward(self, u, saliency_attention_weights=None):
        batch_size = u.size(0)
        u_hat = torch.einsum('bnd, ndkF -> bnkF', u, self.W)
        b_ij = torch.zeros(batch_size, self.in_capsules, self.out_capsules, device=u.device)

        # Apply Saliency Modulation
        if saliency_attention_weights is not None:
            b_ij *= saliency_attention_weights

        v_j: torch.Tensor = torch.zeros()

        # Dynamic Routing Iterations
        for i in range(self.iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = torch.einsum('bni, bnik -> bik', c_ij, u_hat)
            v_j = squash(s_j, dim=-1)

            if i < self.iterations - 1:
                agreement = torch.einsum('bnik, bik -> bni', u_hat, v_j.detach())
                b_ij += agreement

        return v_j

