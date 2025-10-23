import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vit_b_16, ViT_B_16_Weights

def squash(s, dim=-1):
    """
    Standard Capsule Network squash function.
    The length of the vector represents the probability of existence.
    """
    squared_norm = torch.sum(torch.pow(s, 2), dim=dim, keepdim=True)
    norm = torch.sqrt(squared_norm)
    # Scaled by length to maintain direction and squash magnitude
    return (squared_norm / (1.0 + squared_norm)) * (s / norm)

class PrimaryCapsLayer(nn.Module):
    """
    Convert ViT tokens (spatial features) into Primary Capsules.
    Each ViT token corresponds to a spatial location, which is transformed
    into a capsule vector.
    """
    def __init__(self, vit_dim, num_capsules, capsule_dim):
        super(PrimaryCapsLayer, self).__init__()

        # Map ViT dimension to (num_capsules * capsule_dim)
        self.capsule_dim_total: int = num_capsules * capsule_dim
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        # Linear layer to transform each feature token into a capsule
        self.linear = nn.Sequential(
            nn.Linear(vit_dim, self.capsule_dim_total),
            # Sigmoid ensures activation magnitude is 0-1
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, N_tokens, D_vit).
        # N_tokens excludes the CLS token.
        x = self.linear(x)

        # Reshape to (B, N_tokens * num_capsules, capsule_dim)
        x = x.view(x.size(0), -1, self.capsule_dim)
        return x


class SaliencyGuidedRouting(nn.Module):
    """
    Implements Dynamic Routing with Saliency-Guided Attention Modulation (SGAR).
    Attention weights modulate the initial routing logits b_ij.
    """
    def __init__(self, in_capsules, out_capsules, in_dim, out_dim, iterations=3):
        super(SaliencyGuidedRouting, self).__init__()

        self.in_capsules: int = in_capsules # N_patches * N_capsules_per_patch
        self.out_capsules: int = out_capsules # 7 classes (Digit capsules)
        self.iterations = iterations
        self.W = nn.Parameter(
            torch.randn(in_capsules, out_capsules, in_dim, out_dim),
        )

    def forward(self, u, saliency_attention_weights=None):
        # u shape: (B, in_capsules, in_dim)
        batch_size = u.size(0)

        # Transform Primary Capsules(u) to prediction vectors (u_hat)
        # u_hat shape: (B, in_capsules, out_capsules, out_dim)
        u_hat = torch.einsum('bnd, ndkF -> bnkF', u, self.W)

        # Initialize routing logits b_ij.
        # Shape: (B, in_capsules, out_capsules)
        b_ij = torch.zeros(batch_size, self.in_capsules, self.out_capsules, device=u.device)

        # Apply Saliency Modulation
        if saliency_attention_weights is not None:
            # The routing logits b_ij are modulated by the saliency-derived attention [1]
            # We apply the attention weights to the initial state of b_ij (before iteration 1)
            # This biases the routing towards features derived from salient regions (high weights)
            # saliency_attention_weights shape: (B, in_capsules, 1)
            b_ij *= saliency_attention_weights

        v_j: torch.Tensor = torch.zeros()

        # Dynamic Routing
        for i in range(self.iterations):
            # Routing softmax to get coupling coefficients c_ij
            c_ij = F.softmax(b_ij, dim=2)

            # Weighted sum (s_j): s_j = sum(c_ij * u_hat_j[i])
            # s_j shape: (B, out_capsules, out_dim)
            s_j = torch.einsum('bni, bnik -> bik', c_ij, u_hat)

            # Squash activation (v_j)
            v_j = squash(s_j, dim=-1)

            if i < self.iterations - 1:
                # Update routing logits b_ij (routing-by-agreement)
                # b_ij = b_ij + u_hat * v_j (agreement)
                # u_hat shape: (B, in_capsules, out_capsules, out_dim)
                # v_j shape: (B, out_capsules, out_dim)
                agreement = torch.einsum('bnik, bik -> bni', u_hat, v_j.detach())
                b_ij += agreement

        # Final digit capsules (7 classes)
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

        # Calculate number of tokens excluding CLS token
        # For a 256x256 image with 16x16 patches: N_tokens = (256 / 16) ** 2 = 16 * 16 = 256
        self.n_spatial_tokens = (self.input_res // patch_size) ** 2

        # The PrimaryCapsLayer outputs N_spatial_tokens * N_spatial_tokens (65536) capsules.
        # To simplify, we treat the spatial tokens (256) as the Primary Capsules
        self.n_primary_caps = self.n_spatial_tokens

        self.primary_caps = PrimaryCapsLayer(
            vit_dim=vit_dim,
            num_capsules=1,
            capsule_dim=primary_caps_dim
        )

        # Digit Capsules Layer with SGAR
        self.digit_caps = SaliencyGuidedRouting(
            in_capsules=self.n_primary_caps,
            out_capsules=num_classes,
            in_dim=primary_caps_dim,
            out_dim=digit_caps_dim,
        )

    def _save_gradient(self, grad):
        """Hook for saving the gradient during the backward pass."""
        self.gradients = grad

    def _save_activation(self, module, input, output):
        """Hook for saving the activation during the forward pass."""
        # Output of ViT's ln_2 is the processed sequence of tokens
        self.activations = output[:, 1:]  # Exclude the CLS token [6]

    def _calculate_saliency_attention(self):
        """
        Calculates Grad-CAM++ map based on captured gradients and activations.
        """
        if self.gradients is None or self.activations is None:
            # Fallback if hooks were not properly triggered (e.g., during inference without loss)
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
        digit_caps = self.digit_caps(primary_caps, saliency_attention_weights)
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
        num_classes = input_probs.size(1)
        T_c = torch.eye(num_classes, device=target.device)[target]

        loss_present = T_c * F.relu(self.m_plus - input_probs) ** 2
        loss_absent = (1.0 - T_c) * F.relu(input_probs - self.m_minus) ** 2
        L_margin = torch.sum(loss_present + 0.5 * loss_absent, dim=1).mean()

        # Cross Entrop Regularization Component
        L_ce = self.ce_loss(input_probs, target)

        L_total = L_margin + self.lambda_reg * L_ce
        return L_total

