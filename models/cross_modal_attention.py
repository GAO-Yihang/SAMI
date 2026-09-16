import math
from typing import List, Optional

import torch
import torch.nn as nn


def _build_neighbor_cache(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> List[List[int]]:
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]

    if edge_index.numel() == 0:
        return neighbors

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        if d not in neighbors[s]:
            neighbors[s].append(d)

    return neighbors


class SpatialCrossModalAttention(nn.Module):
    """
    Bidirectional neighborhood cross-modal attention.

    - omics1 query attends over omics2 neighborhood (self + 1-hop neighbors)
    - omics2 query attends over omics1 neighborhood (self + 1-hop neighbors)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_omics1 = nn.Linear(hidden_dim, hidden_dim)
        self.k_omics2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_omics2 = nn.Linear(hidden_dim, hidden_dim)
        self.o_omics1 = nn.Linear(hidden_dim, hidden_dim)

        self.q_omics2 = nn.Linear(hidden_dim, hidden_dim)
        self.k_omics1 = nn.Linear(hidden_dim, hidden_dim)
        self.v_omics1 = nn.Linear(hidden_dim, hidden_dim)
        self.o_omics2 = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.num_heads, self.head_dim)

    def _single_direction_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neighbors: List[List[int]],
    ) -> torch.Tensor:
        num_nodes = q.shape[0]
        q_h = self._reshape_heads(q)
        k_h = self._reshape_heads(k)
        v_h = self._reshape_heads(v)

        out = torch.zeros((num_nodes, self.num_heads, self.head_dim), device=q.device, dtype=q.dtype)

        for i in range(num_nodes):
            cand = [i] + neighbors[i]
            idx = torch.tensor(cand, dtype=torch.long, device=q.device)

            qi = q_h[i]
            kj = k_h[idx]
            vj = v_h[idx]

            scores = (qi.unsqueeze(0) * kj).sum(dim=-1).transpose(0, 1) / math.sqrt(self.head_dim)

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_i = torch.einsum("hm,mhd->hd", attn, vj)
            out[i] = out_i

        return out.reshape(num_nodes, self.hidden_dim)

    def forward(
        self,
        omics1_feat: torch.Tensor,
        omics2_feat: torch.Tensor,
        edge_index: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ):
        del coords
        num_nodes = omics1_feat.shape[0]
        neighbors = _build_neighbor_cache(edge_index, num_nodes)

        omics1_to_omics2 = self._single_direction_attention(
            q=self.q_omics1(omics1_feat),
            k=self.k_omics2(omics2_feat),
            v=self.v_omics2(omics2_feat),
            neighbors=neighbors,
        )
        omics2_to_omics1 = self._single_direction_attention(
            q=self.q_omics2(omics2_feat),
            k=self.k_omics1(omics1_feat),
            v=self.v_omics1(omics1_feat),
            neighbors=neighbors,
        )

        omics1_fused = omics1_feat + self.o_omics1(omics1_to_omics2)
        omics2_fused = omics2_feat + self.o_omics2(omics2_to_omics1)
        return omics1_fused, omics2_fused
