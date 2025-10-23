import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
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

class SGViTCapsNet(nn.Module):
    """
    Saliency-Guided Vision Transformer-Capsule Network
    """
    def __init__(self, num_classes=7, patch_size=16, primary_caps_dim=16, digit_caps_dim=16):
        super(SGViTCapsNet, self).__init__()

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.input_res = 256

        self.vit = vit_b_16(pretrained=True, weights=ViT_B_16_Weights.DEFAULT)
        vit_dim = self.vit.hidden_size

        # Grad-CAM specifics
        self.target_layer = self.vit.encoder.layer[-1].ln_2
        self.gradients = None
        self.activations = None

        self.n_spatial_tokens = (self.input_res // patch_size) ** 2
        self.n_primary_caps = self.n_spatial_tokens

        self.primary_caps = PrimaryCapsLayer(
            vit_dim=vit_dim,
            num_capsules=1,
            capsule_dim=primary_caps_dim
        )

        self.digit_caps = SaliencyGuidedRouting(
            in_capsules=self.n_primary_caps,
            out_capsules=num_classes,
            in_dim=primary_caps_dim,
            out_dim=digit_caps_dim,
        )

    def _save_gradient(self, grad):
        self.gradients = grad

    def _save_activation(self, module, input, output):
        self.activations = output[:, 1:]

    def _calculate_saliency_attention(self):
        """
        Calculates Grad-CAM++ map based on captured gradients and activations.
        """
        if self.gradients is None or self.activations is None:
            B = self.activations.size(0) if self.activations is not None else 1
            device = self.activations.device if self.activations is not None else 'cpu'
            return torch.ones(B, self.n_primary_caps, 1, device=device)

        # Element-wise ReLU of gradients (G+), i.e. contribution of positive gradients
        # gradients_relu shape: (B, N_tokens, D_ViT)
        gradients_relu = torch.relu(self.gradients)

        # Calculate Token/Channel Contribution Numerator (A * A * G+)
        # Approximation of the higher-order interaction in Grad-CAM++
        # I_token shape: (B, N_tokens, D_ViT)
        I_token = (self.activations ** 2) * gradients_relu

        # Calculate Normalization Factor Denominator (A * G+)
        # Sum of Activations * Gradient+ across all spatial tokens
        # denominator shape: (B, D_ViT)
        denominator = torch.sum(self.activations * gradients_relu, dim=1) + 1e-8

        # Calculate Alpha (Channel Importance weight)
        # Sum of the token importance (I_token) across all spatial tokens (N_tokens)
        # numerator_sum shape: (B, D_ViT)
        numerator_sum = torch.sum(I_token, dim=1)

        # Final Alpha_k calculation (weighted average of gradients)
        alpha_k = numerator_sum / denominator

        # Weighted Feature Map calculation
        # Weighted Activation = $ Sum_{k} alpha_k * Activation_k $
        weighted_activation = self.activations * alpha_k.unsqueeze(1)

        # Saliency Map: ReLU on summed map
        # Saliency map raw shape: (B, N_tokens)
        saliency_map_raw = torch.relu(torch.sum(weighted_activation, dim=-1))

        # Normalize map to be used as attention weights
        B, N_tokens = saliency_map_raw.shape
        min_val, _ = saliency_map_raw.min(dim=1, keepdim=True)
        max_val, _ = saliency_map_raw.max(dim=1, keepdim=True)

        saliency_attention = (saliency_map_raw - min_val) / (max_val - min_val + 1e-8)

        self.gradients = None
        self.activations = None

        return saliency_attention.unsqueeze(-1)

    def forward(self, x, saliency_attention_weights=None):
        h_a = self.target_layer.register_forward_hook(self._save_activation)

        tokens_out = self.vit.forward_features(x)
        h_a.remove()

        tokens_for_caps = self.activations

        if tokens_for_caps is None:
            raise RuntimeError("Activations not captured by ViT hook. Check target layer.")

        primary_caps = self.primary_caps(tokens_for_caps)
        digit_caps = self.digit_caps(tokens_for_caps, saliency_attention_weights)
        probs = digit_caps.norm(dim=-1)

        return probs, digit_caps


class CapsuleHybridLoss(nn.Module):
    """
    Combines CapsNet Margin Loss and standard Cross Entropy Loss.
    """
    def __init__(self, lambda_reg=0.1, m_plus=0.9, m_minus=0.1):
        super(CapsuleHybridLoss, self).__init__()

        self.lambda_reg = lambda_reg
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input_probs, input_digit_caps, target):
        # Margin Loss component
        num_classes = input_probs.size(-1)
        T_c = torch.eye(num_classes, device=target.device)[target]

        loss_present = T_c * F.relu(self.m_plus - input_probs) ** 2
        loss_absent = (1.0 - T_c) * F.relu(input_probs - self.m_minus) ** 2
        L_margin = torch.sum(loss_present + 0.5 * loss_absent, dim=1).mean()

        # Cross Entrop Regularization Component
        L_ce = self.ce_loss(input_probs, target)

        L_total = L_margin + self.lambda_reg * L_ce
        return L_total

