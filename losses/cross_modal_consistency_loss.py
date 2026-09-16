from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalConsistencyLoss(nn.Module):
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

        self.omics1_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.projector_hidden_dim),
            nn.GELU(),
            nn.Linear(self.projector_hidden_dim, self.projector_out_dim),
        )
        self.omics2_projector = nn.Sequential(
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
        fused_emb: torch.Tensor,
    ) -> None:
        if omics1_emb.ndim != 2 or omics2_emb.ndim != 2 or fused_emb.ndim != 2:
            raise ValueError(
                "omics1_emb, omics2_emb, and fused_emb must be 2D tensors, "
                f"got {tuple(omics1_emb.shape)}, {tuple(omics2_emb.shape)}, "
                f"and {tuple(fused_emb.shape)}"
            )
        if omics1_emb.shape != omics2_emb.shape:
            raise ValueError(
                "omics1_emb and omics2_emb must have the same shape, "
                f"got {tuple(omics1_emb.shape)} and {tuple(omics2_emb.shape)}"
            )
        if omics1_emb.shape[0] != fused_emb.shape[0]:
            raise ValueError(
                "omics embeddings and fused_emb must have the same number of spots, "
                f"got {omics1_emb.shape[0]} and {fused_emb.shape[0]}"
            )
        if omics1_emb.shape[1] != self.hidden_dim:
            raise ValueError(
                f"embedding dim {omics1_emb.shape[1]} does not match hidden_dim {self.hidden_dim}"
            )
        if fused_emb.shape[1] != self.projector_out_dim:
            raise ValueError(
                "fused_emb dim must match projector_out_dim when no fused projector is used, "
                f"got fused_emb dim {fused_emb.shape[1]} and projector_out_dim {self.projector_out_dim}. "
                "Set loss.consistency.projector_out_dim to model.hidden_dim."
            )

    def _project(
        self,
        omics1_emb: torch.Tensor,
        omics2_emb: torch.Tensor,
        fused_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        omics1_proj = F.normalize(self.omics1_projector(omics1_emb), p=2, dim=-1)
        omics2_proj = F.normalize(self.omics2_projector(omics2_emb), p=2, dim=-1)
        fused_proj = F.normalize(fused_emb, p=2, dim=-1)
        return omics1_proj, omics2_proj, fused_proj

    def _pair_loss(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        logits = left @ right.transpose(0, 1)
        logits = logits / self.temperature
        target = torch.arange(left.shape[0], device=left.device)
        return F.cross_entropy(logits, target)

    def forward(
        self,
        omics1_emb: torch.Tensor,
        omics2_emb: torch.Tensor,
        fused_emb: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(omics1_emb, omics2_emb, fused_emb)
        if omics1_emb.shape[0] <= 1:
            return self._zero_loss(omics1_emb, omics2_emb, fused_emb)

        omics1_proj, omics2_proj, fused_proj = self._project(
            omics1_emb,
            omics2_emb,
            fused_emb,
        )
        losses = [
            self._pair_loss(omics1_proj, omics2_proj),
            self._pair_loss(omics2_proj, omics1_proj),
            self._pair_loss(omics1_proj, fused_proj),
            self._pair_loss(fused_proj, omics1_proj),
            self._pair_loss(omics2_proj, fused_proj),
            self._pair_loss(fused_proj, omics2_proj),
        ]
        return torch.stack(losses).mean()
