import torch
import torch.nn as nn

from squash import squash


class ClassCapsule(nn.Module):
    """The class capsules layer which performs routing-by-agreement."""
    def __init__(
            self,
            num_capsules: int,
            num_routes: int,
            in_dim: int,
            out_dim: int,
            num_routing: int = 3,
    ) -> None:
        super().__init__()
        self.num_routing: int = num_routing
        self.vote: torch.Tensor = nn.Parameter(
            torch.randn(
                1,
                num_routes,
                num_capsules,
                out_dim,
                in_dim
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2).unsqueeze(3)
        votes: torch.Tensor = torch.matmul(self.vote, x)
        agreement: torch.Tensor = torch.zeros_like(votes)
        verdict: torch.Tensor = torch.Tensor()

        for _ in range(self.num_routing):
            chairman: torch.Tensor = torch.softmax(agreement, dim=1)
            summary: torch.Tensor = (chairman * votes).sum(dim=1, keepdim=True)
            verdict = squash(summary)
            agreement += (votes * verdict).sum(dim=3, keepdim=True)

        return verdict.squeeze(1).squeeze(-1)