from typing import Tuple

import torch


def neighbor_mean(
    feat: torch.Tensor,
    edge_index: torch.Tensor,
    exclude_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if feat.ndim != 2:
        raise ValueError(f"feat must be [N, D], got {tuple(feat.shape)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")

    num_nodes = feat.shape[0]
    neighbor_mean = torch.zeros_like(feat)
    valid_mask = torch.zeros((num_nodes,), dtype=torch.bool, device=feat.device)

    if edge_index.numel() == 0:
        return neighbor_mean, valid_mask

    src = edge_index[0]
    dst = edge_index[1]

    if exclude_self:
        valid_edges = src != dst
        src = src[valid_edges]
        dst = dst[valid_edges]

    if src.numel() == 0:
        return neighbor_mean, valid_mask

    neighbor_sum = torch.zeros_like(feat)
    neighbor_sum.index_add_(0, src, feat[dst])

    neighbor_count = torch.zeros((num_nodes,), dtype=feat.dtype, device=feat.device)
    neighbor_count.index_add_(0, src, torch.ones_like(src, dtype=feat.dtype))

    valid_mask = neighbor_count > 0
    if valid_mask.any():
        neighbor_mean[valid_mask] = (
            neighbor_sum[valid_mask] / neighbor_count[valid_mask].unsqueeze(1)
        )

    return neighbor_mean, valid_mask
