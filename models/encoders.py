from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    Adapter for high-dimensional pre-extracted features.

    Supported adapter types:
    - identity: pass-through when in_dim == hidden_dim
    - linear: single Linear projection
    - mlp_bottleneck: Linear -> LayerNorm -> GELU -> Dropout -> Linear
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        adapter_type: str = "mlp_bottleneck",
        adapter_dim: Optional[int] = None,
        dropout: float = 0.1,
        force_linear: bool = False,
    ) -> None:
        super().__init__()

        if force_linear:
            adapter_type = "linear"

        if adapter_type not in {"identity", "linear", "mlp_bottleneck"}:
            raise ValueError(
                "adapter_type must be one of: identity, linear, mlp_bottleneck"
            )

        if adapter_type == "identity":
            if in_dim != hidden_dim:
                raise ValueError(
                    "identity adapter requires in_dim == hidden_dim, "
                    f"got {in_dim} and {hidden_dim}"
                )
            self.proj = nn.Identity()
        elif adapter_type == "linear":
            self.proj = nn.Linear(in_dim, hidden_dim)
        else:
            if in_dim == hidden_dim:
                self.proj = nn.Identity()
            else:
                if adapter_dim is None:
                    raise ValueError("adapter_dim must be provided for mlp_bottleneck adapter")
                self.proj = nn.Sequential(
                    nn.Linear(in_dim, adapter_dim),
                    nn.LayerNorm(adapter_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(adapter_dim, hidden_dim),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
class PointNeXtLocalAggregation(nn.Module):
    """
    Fixed-graph PointNeXt-style local aggregation.

    Aggregates expanded features over the existing spot graph without sampling.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
        pooling: str = "max",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if pooling not in {"max", "mean"}:
            raise ValueError("pooling must be one of: max, mean")

        local_in_dim = channels * 2 + 2
        self.pooling = pooling
        self.local_mlp = nn.Sequential(
            nn.Linear(local_in_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(channels)

    def _pool(self, local_out: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        mask = neighbor_mask.unsqueeze(-1)
        if self.pooling == "max":
            masked = local_out.masked_fill(~mask, torch.finfo(local_out.dtype).min)
            return masked.max(dim=1).values

        weight = mask.to(dtype=local_out.dtype)
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (local_out * weight).sum(dim=1) / denom

    def forward(
        self,
        h: torch.Tensor,
        coords_norm: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        if h.ndim != 2:
            raise ValueError(f"h must be [N, C], got {tuple(h.shape)}")

        center_feat = h.unsqueeze(1).expand(-1, neighbor_idx.shape[1], -1)
        neighbor_feat = h[neighbor_idx]
        delta_feat = neighbor_feat - center_feat

        center_xy = coords_norm.unsqueeze(1).expand(-1, neighbor_idx.shape[1], -1)
        neighbor_xy = coords_norm[neighbor_idx]
        delta_xy = neighbor_xy - center_xy

        local_input = torch.cat([center_feat, delta_feat, delta_xy], dim=-1)
        local_out = self.local_mlp(local_input)
        pooled = self._pool(local_out, neighbor_mask)
        return self.out_norm(pooled)


class PointNeXtBlock(nn.Module):
    """
    Fixed-width PointNeXt-style residual block.

    pre_norm -> up_proj -> local aggregation -> down_proj -> residual -> post_norm
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        dropout: float = 0.1,
        pooling: str = "max",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")

        expanded_dim = hidden_dim * expansion
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.up_proj = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
        )
        self.local_agg = PointNeXtLocalAggregation(
            channels=expanded_dim,
            dropout=dropout,
            pooling=pooling,
        )
        self.down_proj = nn.Sequential(
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.post_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        coords_norm: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must be [N, D], got {tuple(x.shape)}")

        shortcut = x
        h = self.pre_norm(x)
        h = self.up_proj(h)
        h = self.local_agg(h, coords_norm, neighbor_idx, neighbor_mask)
        h = self.down_proj(h)
        return self.post_norm(shortcut + h)


class PointNeXtGraphBackbone(nn.Module):
    """
    Fixed-graph PointNeXt backbone.

    Uses the existing spot graph, keeps the number of spots unchanged,
    and returns stage-wise hidden representations.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        expansion: int = 4,
        dropout: float = 0.1,
        max_neighbors: int = 6,
        pooling: str = "max",
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if max_neighbors <= 0:
            raise ValueError(f"max_neighbors must be positive, got {max_neighbors}")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.expansion = int(expansion)
        self.max_neighbors = int(max_neighbors)
        self.blocks = nn.ModuleList(
            [
                PointNeXtBlock(
                    hidden_dim=hidden_dim,
                    expansion=expansion,
                    dropout=dropout,
                    pooling=pooling,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _normalize_coords(coords: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be [N, 2], got {tuple(coords.shape)}")
        mu = coords.mean(dim=0, keepdim=True)
        sigma = coords.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        return (coords - mu) / sigma

    @staticmethod
    def _sorted_neighbors(
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> List[List[int]]:
        neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
        if edge_index.numel() == 0:
            return neighbors

        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            if 0 <= s < num_nodes and 0 <= d < num_nodes:
                neighbors[s].append(int(d))
        return neighbors

    def _build_neighbor_tensors(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sorted_neighbors = self._sorted_neighbors(edge_index, num_nodes)
        neighbor_idx = torch.zeros((num_nodes, self.max_neighbors), dtype=torch.long, device=device)
        neighbor_mask = torch.zeros((num_nodes, self.max_neighbors), dtype=torch.bool, device=device)

        for node in range(num_nodes):
            items = sorted_neighbors[node][: self.max_neighbors]
            if not items:
                items = [node]

            for pos, nbr in enumerate(items):
                neighbor_idx[node, pos] = nbr
                neighbor_mask[node, pos] = True

            if len(items) < self.max_neighbors:
                neighbor_idx[node, len(items) :] = node

        return neighbor_idx, neighbor_mask

    def forward(
        self,
        coords: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_stage_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if x.ndim != 2:
            raise ValueError(f"x must be [N, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"x last dim must match hidden_dim {self.hidden_dim}, got {x.shape[-1]}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")

        coords_norm = self._normalize_coords(coords)
        neighbor_idx, neighbor_mask = self._build_neighbor_tensors(
            num_nodes=x.shape[0],
            edge_index=edge_index,
            device=x.device,
        )

        stage_outputs: List[torch.Tensor] = []
        h = x
        for block in self.blocks:
            h = block(h, coords_norm, neighbor_idx, neighbor_mask)
            stage_outputs.append(h)

        if return_stage_outputs:
            return h, stage_outputs
        return h
