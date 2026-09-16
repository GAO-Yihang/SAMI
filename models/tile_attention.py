from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import softmax


class VectorizedSpatialCrossModalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.q_omics1 = nn.Linear(hidden_dim, hidden_dim)
        self.k_omics2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_omics2 = nn.Linear(hidden_dim, hidden_dim)
        self.o_omics1 = nn.Linear(hidden_dim, hidden_dim)
        self.q_omics2 = nn.Linear(hidden_dim, hidden_dim)
        self.k_omics1 = nn.Linear(hidden_dim, hidden_dim)
        self.v_omics1 = nn.Linear(hidden_dim, hidden_dim)
        self.o_omics2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(value.shape[0], self.num_heads, self.head_dim)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = query.shape[0]
        loops = torch.arange(num_nodes, device=edge_index.device)
        source = torch.cat([edge_index[0], loops])
        neighbor = torch.cat([edge_index[1], loops])
        query_heads = self._heads(query)
        key_heads = self._heads(key)
        value_heads = self._heads(value)
        scores = (
            query_heads[source] * key_heads[neighbor]
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        weights = softmax(scores, source, num_nodes=num_nodes)
        weights = self.dropout(weights).to(dtype=value_heads.dtype)
        messages = weights.unsqueeze(-1) * value_heads[neighbor]
        output = torch.zeros(
            (num_nodes, self.num_heads, self.head_dim),
            device=query.device,
            dtype=query.dtype,
        )
        output.index_add_(0, source, messages)
        return output.reshape(num_nodes, self.hidden_dim)

    def forward(
        self,
        omics1_feat: torch.Tensor,
        omics2_feat: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del coords
        omics1_delta = self._attend(
            self.q_omics1(omics1_feat),
            self.k_omics2(omics2_feat),
            self.v_omics2(omics2_feat),
            edge_index,
        )
        omics2_delta = self._attend(
            self.q_omics2(omics2_feat),
            self.k_omics1(omics1_feat),
            self.v_omics1(omics1_feat),
            edge_index,
        )
        return (
            omics1_feat + self.o_omics1(omics1_delta),
            omics2_feat + self.o_omics2(omics2_delta),
        )


class VectorizedSpatialTriModalAttention(nn.Module):
    branches = ("omics1", "omics2", "omics3")

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.query = nn.ModuleDict(
            {branch: nn.Linear(hidden_dim, hidden_dim) for branch in self.branches}
        )
        self.key = nn.ModuleDict(
            {branch: nn.Linear(hidden_dim, hidden_dim) for branch in self.branches}
        )
        self.value = nn.ModuleDict(
            {branch: nn.Linear(hidden_dim, hidden_dim) for branch in self.branches}
        )
        self.output = nn.ModuleDict(
            {
                f"{target}_from_{source}": nn.Linear(hidden_dim, hidden_dim)
                for target in self.branches
                for source in self.branches
                if target != source
            }
        )
        self.dropout = nn.Dropout(dropout)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(value.shape[0], self.num_heads, self.head_dim)

    def _pair_delta(
        self,
        target: str,
        source_name: str,
        target_feat: torch.Tensor,
        source_feat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        query = self._heads(self.query[target](target_feat))
        source_key = self._heads(self.key[source_name](source_feat))
        source_value = self._heads(self.value[source_name](source_feat))
        num_nodes = target_feat.shape[0]
        loops = torch.arange(num_nodes, device=edge_index.device)
        anchors = torch.cat([edge_index[0], loops])
        neighbors = torch.cat([edge_index[1], loops])
        scores = (
            query[anchors] * source_key[neighbors]
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        weights = self.dropout(
            softmax(scores, anchors, num_nodes=num_nodes)
        ).to(dtype=source_value.dtype)
        messages = weights.unsqueeze(-1) * source_value[neighbors]
        output = torch.zeros(
            (num_nodes, self.num_heads, self.head_dim),
            device=target_feat.device,
            dtype=target_feat.dtype,
        )
        output.index_add_(0, anchors, messages)
        return self.output[f"{target}_from_{source_name}"](
            output.reshape(num_nodes, self.hidden_dim)
        )

    def forward(
        self,
        omics1_feat: torch.Tensor,
        omics2_feat: torch.Tensor,
        omics3_feat: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del coords
        features = {
            "omics1": omics1_feat,
            "omics2": omics2_feat,
            "omics3": omics3_feat,
        }
        fused: Dict[str, torch.Tensor] = {}
        for target in self.branches:
            sources = [branch for branch in self.branches if branch != target]
            deltas = [
                self._pair_delta(
                    target,
                    source,
                    features[target],
                    features[source],
                    edge_index,
                )
                for source in sources
            ]
            fused[target] = features[target] + 0.5 * (deltas[0] + deltas[1])
        return fused["omics1"], fused["omics2"], fused["omics3"]
