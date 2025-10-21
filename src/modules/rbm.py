from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    def __init__(
            self,
            n_visible: int,
            n_hidden: int,
            k: int = 1
    ) -> None:
        super(RBM, self).__init__()
        self.W: torch.Tensor = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.hidden_bias: torch.Tensor = nn.Parameter(torch.zeros(n_hidden))
        self.visible_bias: torch.Tensor = nn.Parameter(torch.zeros(n_visible))
        self.k = k

    def sample_hidden(self, visible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prob = torch.sigmoid(F.linear(visible, self.W, self.hidden_bias))
        return h_prob, torch.bernoulli(h_prob)

    def sample_visible(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_prob = torch.sigmoid(F.linear(hidden, self.W.t(), self.visible_bias))
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, visible: torch.Tensor) -> torch.Tensor:
        _, h_sample_0 = self.sample_visible(visible)
        h_sample_k: torch.Tensor = h_sample_0
        v_prob_k: torch.Tensor = torch.Tensor()

        for _ in range(self.k):
            v_prob_k, _ = self.sample_visible(h_sample_k)
            _, h_sample_k = self.sample_hidden(v_prob_k)

        return torch.mean((visible - v_prob_k) ** 2)

    def pass_through(self, visible: torch.Tensor) -> torch.Tensor:
        return self.sample_hidden(visible)[0]
