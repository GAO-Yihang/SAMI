from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphContrastiveLoss(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_symmetric_loss: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.hidden_dim = int(hidden_dim)
        self.use_symmetric_loss = bool(use_symmetric_loss)
        self.weight = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def _build_local_context(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError(f"z must be 2D, got shape {tuple(z.shape)}")
        if z.shape[1] != self.hidden_dim:
            raise ValueError(
                f"z hidden dim {z.shape[1]} does not match loss hidden_dim {self.hidden_dim}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
            )

        num_nodes = z.shape[0]
        contexts = torch.zeros_like(z)
        has_neighbor = torch.zeros((num_nodes,), dtype=torch.bool, device=z.device)
        if edge_index.numel() == 0:
            return contexts, has_neighbor

        src = edge_index[0]
        dst = edge_index[1]
        valid = src != dst
        src = src[valid]
        dst = dst[valid]

        if src.numel() == 0:
            return contexts, has_neighbor

        neighbor_sum = torch.zeros_like(z)
        neighbor_sum.index_add_(0, src, z[dst])

        neighbor_count = torch.zeros((num_nodes,), dtype=z.dtype, device=z.device)
        neighbor_count.index_add_(0, src, torch.ones_like(src, dtype=z.dtype))

        has_neighbor = neighbor_count > 0
        contexts[has_neighbor] = (
            neighbor_sum[has_neighbor] / neighbor_count[has_neighbor].unsqueeze(1)
        )
        contexts = torch.sigmoid(contexts)
        return contexts, has_neighbor

    def _score(
        self,
        anchor: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        projected_anchor = anchor @ self.weight
        return (projected_anchor * context).sum(dim=-1) + self.bias

    @staticmethod
    def _zero_loss(z: torch.Tensor, z_corrupt: torch.Tensor) -> torch.Tensor:
        return (z.sum() + z_corrupt.sum()) * 0.0

    def forward(
        self,
        z: torch.Tensor,
        z_corrupt: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if z.shape != z_corrupt.shape:
            raise ValueError(
                f"z and z_corrupt must have the same shape, got {tuple(z.shape)} and "
                f"{tuple(z_corrupt.shape)}"
            )

        num_nodes = z.shape[0]
        if num_nodes <= 1:
            return self._zero_loss(z, z_corrupt)

        context, has_neighbor = self._build_local_context(z, edge_index)
        context_corrupt, has_neighbor_corrupt = self._build_local_context(z_corrupt, edge_index)
        valid_anchor = has_neighbor & has_neighbor_corrupt

        if not valid_anchor.any():
            return self._zero_loss(z, z_corrupt)

        pos_logits = self._score(z, context)[valid_anchor]
        neg_logits = self._score(z_corrupt, context)[valid_anchor]
        loss_pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        loss_neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = 0.5 * (loss_pos + loss_neg)

        if self.use_symmetric_loss:
            pos_logits_corrupt = self._score(z_corrupt, context_corrupt)[valid_anchor]
            neg_logits_corrupt = self._score(z, context_corrupt)[valid_anchor]
            loss_pos_corrupt = F.binary_cross_entropy_with_logits(
                pos_logits_corrupt,
                torch.ones_like(pos_logits_corrupt),
            )
            loss_neg_corrupt = F.binary_cross_entropy_with_logits(
                neg_logits_corrupt,
                torch.zeros_like(neg_logits_corrupt),
            )
            loss = loss + 0.5 * (loss_pos_corrupt + loss_neg_corrupt)

        return loss