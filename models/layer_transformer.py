from typing import List

import torch
import torch.nn as nn


class LayerwiseTransformerAggregator(nn.Module):
    """
    Aggregate multi-layer node representations along layer axis.
    Input: List[[N, D]] with length L
    Output: [N, D]
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        self.layer_pos_embed = nn.Parameter(torch.randn(num_layers, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if pooling == "attn":
            self.pool_vec = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        if len(layer_outputs) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer outputs, got {len(layer_outputs)}"
            )

        x = torch.stack(layer_outputs, dim=1)  # [N, L, D]
        x = x + self.layer_pos_embed.unsqueeze(0)
        x = self.encoder(x)

        if self.pooling == "mean":
            return x.mean(dim=1)
        if self.pooling == "last":
            return x[:, -1, :]
        if self.pooling == "attn":
            score = torch.einsum("nld,d->nl", x, self.pool_vec)
            w = torch.softmax(score, dim=-1)
            return torch.einsum("nl,nld->nd", w, x)

        raise ValueError(f"Unsupported pooling type: {self.pooling}")