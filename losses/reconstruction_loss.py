import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type: str = "mse") -> None:
        super().__init__()
        if loss_type != "mse":
            raise ValueError("Baseline only supports mse reconstruction loss")
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)