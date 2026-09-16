import torch
import torch.nn as nn


class ConcatMLPFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(self, omics1_emb: torch.Tensor, omics2_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([omics1_emb, omics2_emb], dim=-1)
        return self.mlp(x)


class ConcatMLP3Fusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        omics1_emb: torch.Tensor,
        omics2_emb: torch.Tensor,
        omics3_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([omics1_emb, omics2_emb, omics3_emb], dim=-1)
        return self.mlp(x)
