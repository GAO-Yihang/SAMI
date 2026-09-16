from itertools import combinations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriModalConsistencyLoss(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        temperature: float = 0.1,
        projector_hidden_dim: Optional[int] = None,
        projector_out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.projector_hidden_dim = int(projector_hidden_dim or hidden_dim)
        self.projector_out_dim = int(projector_out_dim or hidden_dim)

        self.projectors = nn.ModuleDict(
            {
                branch: self._make_projector()
                for branch in ("omics1", "omics2", "omics3", "z")
            }
        )

    def _make_projector(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.projector_hidden_dim),
            nn.GELU(),
            nn.Linear(self.projector_hidden_dim, self.projector_out_dim),
        )

    @staticmethod
    def _zero_loss(*tensors: torch.Tensor) -> torch.Tensor:
        return sum(t.sum() for t in tensors) * 0.0

    def _validate(
        self,
        omics1_emb: torch.Tensor,
        omics2_emb: torch.Tensor,
        omics3_emb: torch.Tensor,
        fused_emb: torch.Tensor,
    ) -> None:
        tensors = {
            "omics1_emb": omics1_emb,
            "omics2_emb": omics2_emb,
            "omics3_emb": omics3_emb,
            "fused_emb": fused_emb,
        }
        for name, tensor in tensors.items():
            if tensor.ndim != 2:
                raise ValueError(f"{name} must be 2D, got {tuple(tensor.shape)}")
            if tensor.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"{name} dim {tensor.shape[1]} does not match hidden_dim "
                    f"{self.hidden_dim}"
                )

        n = omics1_emb.shape[0]
        for name, tensor in tensors.items():
            if tensor.shape[0] != n:
                raise ValueError(
                    f"{name} has mismatched number of spots {tensor.shape[0]} vs {n}"
                )

    def _project(
        self,
        omics1_emb: torch.Tensor,
        omics2_emb: torch.Tensor,
        omics3_emb: torch.Tensor,
        fused_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        inputs = {
            "omics1": omics1_emb,
            "omics2": omics2_emb,
            "omics3": omics3_emb,
            "z": fused_emb,
        }
        return {
            name: F.normalize(self.projectors[name](tensor), p=2, dim=-1)
            for name, tensor in inputs.items()
        }

    def _pair_loss(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        logits = left @ right.transpose(0, 1)
        logits = logits / self.temperature
        target = torch.arange(left.shape[0], device=left.device)
        return F.cross_entropy(logits, target)

    def _symmetric_pair_loss(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._pair_loss(left, right), self._pair_loss(right, left)

    def forward(
        self,
        omics1_emb: torch.Tensor,
        omics2_emb: torch.Tensor,
        omics3_emb: torch.Tensor,
        fused_emb: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(omics1_emb, omics2_emb, omics3_emb, fused_emb)
        if omics1_emb.shape[0] <= 1:
            return self._zero_loss(omics1_emb, omics2_emb, omics3_emb, fused_emb)

        projected = self._project(omics1_emb, omics2_emb, omics3_emb, fused_emb)
        losses = []

        for left_name, right_name in combinations(("omics1", "omics2", "omics3"), 2):
            losses.extend(
                self._symmetric_pair_loss(projected[left_name], projected[right_name])
            )

        for branch in ("omics1", "omics2", "omics3"):
            losses.extend(self._symmetric_pair_loss(projected[branch], projected["z"]))

        return torch.stack(losses).mean()
