import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


TRI_MODAL_BRANCHES = ("omics1", "omics2", "omics3")


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
        if 0 <= s < num_nodes and 0 <= d < num_nodes and d not in neighbors[s]:
            neighbors[s].append(int(d))

    return neighbors


class SpatialTriModalAttention(nn.Module):
    """
    Symmetric pairwise neighborhood cross-modal attention for three modalities.

    Each modality queries the 1-hop neighborhood of the other two modalities.
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

        self.query = nn.ModuleDict(
            {branch: nn.Linear(hidden_dim, hidden_dim) for branch in TRI_MODAL_BRANCHES}
        )
        self.key = nn.ModuleDict(
            {branch: nn.Linear(hidden_dim, hidden_dim) for branch in TRI_MODAL_BRANCHES}
        )
        self.value = nn.ModuleDict(
            {branch: nn.Linear(hidden_dim, hidden_dim) for branch in TRI_MODAL_BRANCHES}
        )
        self.output = nn.ModuleDict(
            {
                f"{target}_from_{source}": nn.Linear(hidden_dim, hidden_dim)
                for target in TRI_MODAL_BRANCHES
                for source in TRI_MODAL_BRANCHES
                if target != source
            }
        )
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

        out = torch.zeros(
            (num_nodes, self.num_heads, self.head_dim),
            device=q.device,
            dtype=q.dtype,
        )

        for i in range(num_nodes):
            cand = [i] + neighbors[i]
            idx = torch.tensor(cand, dtype=torch.long, device=q.device)

            qi = q_h[i]
            kj = k_h[idx]
            vj = v_h[idx]

            scores = (qi.unsqueeze(0) * kj).sum(dim=-1).transpose(0, 1)
            scores = scores / math.sqrt(self.head_dim)

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_i = torch.einsum("hm,mhd->hd", attn, vj)
            out[i] = out_i

        return out.reshape(num_nodes, self.hidden_dim)

    def _pair_delta(
        self,
        target: str,
        source: str,
        target_feat: torch.Tensor,
        source_feat: torch.Tensor,
        neighbors: List[List[int]],
    ) -> torch.Tensor:
        attended = self._single_direction_attention(
            q=self.query[target](target_feat),
            k=self.key[source](source_feat),
            v=self.value[source](source_feat),
            neighbors=neighbors,
        )
        return self.output[f"{target}_from_{source}"](attended)

    def forward(
        self,
        omics1_feat: torch.Tensor,
        omics2_feat: torch.Tensor,
        omics3_feat: torch.Tensor,
        edge_index: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ):
        del coords
        feats: Dict[str, torch.Tensor] = {
            "omics1": omics1_feat,
            "omics2": omics2_feat,
            "omics3": omics3_feat,
        }
        num_nodes = omics1_feat.shape[0]
        neighbors = _build_neighbor_cache(edge_index, num_nodes)

        fused: Dict[str, torch.Tensor] = {}
        for target in TRI_MODAL_BRANCHES:
            sources = [branch for branch in TRI_MODAL_BRANCHES if branch != target]
            delta_a = self._pair_delta(
                target=target,
                source=sources[0],
                target_feat=feats[target],
                source_feat=feats[sources[0]],
                neighbors=neighbors,
            )
            delta_b = self._pair_delta(
                target=target,
                source=sources[1],
                target_feat=feats[target],
                source_feat=feats[sources[1]],
                neighbors=neighbors,
            )
            fused[target] = feats[target] + 0.5 * (delta_a + delta_b)

        return fused["omics1"], fused["omics2"], fused["omics3"]
