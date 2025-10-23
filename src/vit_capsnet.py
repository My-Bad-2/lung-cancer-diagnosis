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
