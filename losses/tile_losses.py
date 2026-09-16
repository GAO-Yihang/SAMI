from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.cross_modal_consistency_loss import CrossModalConsistencyLoss
from losses.graph_contrastive_loss import GraphContrastiveLoss
from losses.tri_modal_consistency_loss import TriModalConsistencyLoss


def core_mask_from(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
) -> torch.Tensor:
    mask = batch.get("core_mask", outputs.get("core_mask"))
    if mask is None:
        return torch.ones(outputs["z"].shape[0], dtype=torch.bool, device=outputs["z"].device)
    mask = mask.to(device=outputs["z"].device, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != outputs["z"].shape[0]:
        raise ValueError("core_mask must be one-dimensional and match node count")
    if not mask.any():
        raise ValueError("core_mask contains no core nodes")
    return mask


class MaskedGraphContrastiveLoss(GraphContrastiveLoss):
    def forward(
        self,
        z: torch.Tensor,
        z_corrupt: torch.Tensor,
        edge_index: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if anchor_mask is None:
            return super().forward(z, z_corrupt, edge_index)
        context, has_neighbor = self._build_local_context(z, edge_index)
        context_corrupt, has_neighbor_corrupt = self._build_local_context(
            z_corrupt, edge_index
        )
        valid = (
            anchor_mask.to(device=z.device, dtype=torch.bool)
            & has_neighbor
            & has_neighbor_corrupt
        )
        if not valid.any():
            return self._zero_loss(z, z_corrupt)
        pos_logits = self._score(z, context)[valid]
        neg_logits = self._score(z_corrupt, context)[valid]
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        )
        if self.use_symmetric_loss:
            pos_corrupt = self._score(z_corrupt, context_corrupt)[valid]
            neg_corrupt = self._score(z, context_corrupt)[valid]
            loss = loss + 0.5 * (
                F.binary_cross_entropy_with_logits(
                    pos_corrupt, torch.ones_like(pos_corrupt)
                )
                + F.binary_cross_entropy_with_logits(
                    neg_corrupt, torch.zeros_like(neg_corrupt)
                )
            )
        return loss


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    return F.mse_loss(prediction[mask], target[mask])


def compute_dual_losses(
    config: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    contrastive: MaskedGraphContrastiveLoss,
    consistency: CrossModalConsistencyLoss,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mask = core_mask_from(batch, outputs)
    omics1_target = batch.get("omics1_target", batch["omics1_feat"])
    omics2_target = batch.get("omics2_target", batch["omics2_feat"])
    loss_omics1 = masked_mse(outputs["omics1_recon"], omics1_target, mask)
    loss_omics2 = masked_mse(outputs["omics2_recon"], omics2_target, mask)
    z = outputs["z"]
    z_corrupt = z[torch.randperm(z.shape[0], device=z.device)]
    loss_contrastive = contrastive(z, z_corrupt, outputs["edge_index"], mask)
    loss_consistency = consistency(
        outputs["omics1_multi_scale_embedding"][mask],
        outputs["omics2_multi_scale_embedding"][mask],
        z[mask],
    )
    loss_cfg = config["loss"]
    total = (
        float(loss_cfg["lambda_omics1"]) * loss_omics1
        + float(loss_cfg["lambda_omics2"]) * loss_omics2
        + float(loss_cfg["lambda_contrastive"]) * loss_contrastive
        + float(loss_cfg.get("lambda_consistency", 0.0)) * loss_consistency
    )
    return total, {
        "loss_total": total,
        "loss_omics1": loss_omics1,
        "loss_omics2": loss_omics2,
        "loss_contrastive": loss_contrastive,
        "loss_consistency": loss_consistency,
    }


def compute_trimodal_losses(
    config: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    contrastive: MaskedGraphContrastiveLoss,
    consistency: TriModalConsistencyLoss,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mask = core_mask_from(batch, outputs)
    reconstruction = {}
    for branch in ("omics1", "omics2", "omics3"):
        target = batch.get(f"{branch}_target", batch[f"{branch}_feat"])
        reconstruction[branch] = masked_mse(
            outputs[f"{branch}_recon"], target, mask
        )
    z = outputs["z"]
    z_corrupt = z[torch.randperm(z.shape[0], device=z.device)]
    loss_contrastive = contrastive(z, z_corrupt, outputs["edge_index"], mask)
    loss_consistency = consistency(
        outputs["omics1_multi_scale_embedding"][mask],
        outputs["omics2_multi_scale_embedding"][mask],
        outputs["omics3_multi_scale_embedding"][mask],
        z[mask],
    )
    loss_cfg = config["loss"]
    total = (
        float(loss_cfg["lambda_omics1"]) * reconstruction["omics1"]
        + float(loss_cfg["lambda_omics2"]) * reconstruction["omics2"]
        + float(loss_cfg["lambda_omics3"]) * reconstruction["omics3"]
        + float(loss_cfg["lambda_contrastive"]) * loss_contrastive
        + float(loss_cfg.get("lambda_consistency", 0.0)) * loss_consistency
    )
    return total, {
        "loss_total": total,
        "loss_omics1": reconstruction["omics1"],
        "loss_omics2": reconstruction["omics2"],
        "loss_omics3": reconstruction["omics3"],
        "loss_contrastive": loss_contrastive,
        "loss_consistency": loss_consistency,
    }
