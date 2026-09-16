from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_softmax_per_dst(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    scores: [E, H]
    dst: [E]
    returns normalized scores [E, H]
    """
    out = torch.zeros_like(scores)
    for node in range(num_nodes):
        mask = dst == node
        if not torch.any(mask):
            continue
        out[mask] = torch.softmax(scores[mask], dim=0)
    return out


class GATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.w = nn.Parameter(torch.empty(num_heads, in_dim, self.head_dim))
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))

        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1))
        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)
            if self.res_proj.bias is not None:
                nn.init.zeros_(self.res_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.shape[0]
        if edge_index.numel() == 0:
            return self.res_proj(x)

        src = edge_index[0]
        dst = edge_index[1]

        h = torch.einsum("nd,hdf->nhf", x, self.w)

        h_src = h[src]
        h_dst = h[dst]

        e_src = (h_src * self.a_src.unsqueeze(0)).sum(dim=-1)
        e_dst = (h_dst * self.a_dst.unsqueeze(0)).sum(dim=-1)
        e = self.act(e_src + e_dst)

        alpha = edge_softmax_per_dst(e, dst, num_nodes)
        alpha = self.dropout(alpha)

        msg = alpha.unsqueeze(-1) * h_src
        out = torch.zeros((num_nodes, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        out = out.reshape(num_nodes, self.out_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        out = out + self.res_proj(x)
        return out


class GATStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GATLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index)
            h = F.elu(h)
            h = norm(h)
            outputs.append(h)
        return outputs
