from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from models.full_model import MultimodalSTGraphModel
from models.tri_modal_attention import TRI_MODAL_BRANCHES
from models.tri_modal_full_model import TrimodalSTGraphModel
from utils.graph_utils import neighbor_mean
from models.tile_attention import (
    VectorizedSpatialCrossModalAttention,
    VectorizedSpatialTriModalAttention,
)


def _as_batch(
    batch: Optional[Dict[str, torch.Tensor]],
    kwargs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if batch is not None and kwargs:
        raise ValueError("Pass either a batch mapping or keyword tensors, not both")
    if batch is not None:
        return batch
    return kwargs


class ScalableFMRLModel(MultimodalSTGraphModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        attention_cfg = config["model"]["cross_modal_attention"]
        self.cross_attn = VectorizedSpatialCrossModalAttention(
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_heads=int(attention_cfg["num_heads"]),
            dropout=float(attention_cfg["dropout"]),
        )

    def _apply_graph_backbone(
        self,
        branch: str,
        fused_feat: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        with torch.autocast(device_type=fused_feat.device.type, enabled=False):
            return super()._apply_graph_backbone(
                branch,
                fused_feat.float(),
                coords.float(),
                edge_index,
                outputs,
            )

    def forward(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, Any]:
        batch = _as_batch(batch, kwargs)
        coords = batch["coords"]
        edge_index = batch["edge_index"]
        omics1_feat = batch["omics1_feat"]
        omics2_feat = batch["omics2_feat"]

        if self.omics1_augmentation_enabled:
            neighbor_omics1_feat, _ = neighbor_mean(
                omics1_feat,
                edge_index,
                exclude_self=True,
            )
            omics1_feat_aug = (
                omics1_feat + self.omics1_augmentation_alpha * neighbor_omics1_feat
            )
        else:
            omics1_feat_aug = omics1_feat

        omics1_emb = self.omics1_adapter(omics1_feat_aug)
        omics2_emb = self.omics2_adapter(omics2_feat)
        omics1_fused, omics2_fused = self.cross_attn(
            omics1_feat=omics1_emb,
            omics2_feat=omics2_emb,
            edge_index=edge_index,
            coords=coords,
        )
        outputs: Dict[str, Any] = {
            "branch_mode": self.branch_mode,
            "omics1_feat_aug": omics1_feat_aug,
            "omics1_emb": omics1_emb,
            "omics2_emb": omics2_emb,
            "omics1_fused": omics1_fused,
            "omics2_fused": omics2_fused,
            "edge_index": edge_index,
            "core_mask": batch.get("core_mask"),
        }
        omics1_multi_scale_embedding = self._apply_graph_backbone(
            "omics1", omics1_fused, coords, edge_index, outputs
        )
        omics2_multi_scale_embedding = self._apply_graph_backbone(
            "omics2", omics2_fused, coords, edge_index, outputs
        )
        outputs["omics1_multi_scale_embedding"] = omics1_multi_scale_embedding
        outputs["omics2_multi_scale_embedding"] = omics2_multi_scale_embedding
        if self.fusion is None or self.omics1_decoder is None or self.omics2_decoder is None:
            raise RuntimeError("ScalableFMRLModel requires multimodal_fusion mode")
        z = self.fusion(omics1_multi_scale_embedding, omics2_multi_scale_embedding)
        outputs["z"] = z
        outputs["omics1_recon"] = self.omics1_decoder(z)
        outputs["omics2_recon"] = self.omics2_decoder(z)
        return outputs


class ScalableTriModalFMRLModel(TrimodalSTGraphModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        attention_cfg = config["model"]["cross_modal_attention"]
        self.cross_attn = VectorizedSpatialTriModalAttention(
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_heads=int(attention_cfg["num_heads"]),
            dropout=float(attention_cfg["dropout"]),
        )

    def _apply_graph_backbone(
        self,
        branch: str,
        fused_feat: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        with torch.autocast(device_type=fused_feat.device.type, enabled=False):
            return super()._apply_graph_backbone(
                branch,
                fused_feat.float(),
                coords.float(),
                edge_index,
                outputs,
            )

    def forward(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, Any]:
        batch = _as_batch(batch, kwargs)
        coords = batch["coords"]
        edge_index = batch["edge_index"]
        features = {branch: batch[f"{branch}_feat"] for branch in TRI_MODAL_BRANCHES}
        if self.omics1_augmentation_enabled:
            neighbor_feature, _ = neighbor_mean(
                features["omics1"],
                edge_index,
                exclude_self=True,
            )
            omics1_feat_aug = (
                features["omics1"] + self.omics1_augmentation_alpha * neighbor_feature
            )
        else:
            omics1_feat_aug = features["omics1"]
        features["omics1"] = omics1_feat_aug
        embeddings = {
            branch: self.adapters[branch](features[branch])
            for branch in TRI_MODAL_BRANCHES
        }
        fused_values = self.cross_attn(
            omics1_feat=embeddings["omics1"],
            omics2_feat=embeddings["omics2"],
            omics3_feat=embeddings["omics3"],
            edge_index=edge_index,
            coords=coords,
        )
        fused = dict(zip(TRI_MODAL_BRANCHES, fused_values))
        outputs: Dict[str, Any] = {
            "branch_mode": self.branch_mode,
            "omics1_feat_aug": omics1_feat_aug,
            "edge_index": edge_index,
            "core_mask": batch.get("core_mask"),
        }
        multi_scale_embeddings = {}
        for branch in TRI_MODAL_BRANCHES:
            outputs[f"{branch}_emb"] = embeddings[branch]
            outputs[f"{branch}_fused"] = fused[branch]
            multi_scale_embeddings[branch] = self._apply_graph_backbone(
                branch,
                fused[branch],
                coords,
                edge_index,
                outputs,
            )
            outputs[f"{branch}_multi_scale_embedding"] = multi_scale_embeddings[branch]
        z = self.fusion(
            multi_scale_embeddings["omics1"],
            multi_scale_embeddings["omics2"],
            multi_scale_embeddings["omics3"],
        )
        outputs["z"] = z
        for branch in TRI_MODAL_BRANCHES:
            outputs[f"{branch}_recon"] = self.decoders[branch](z)
        return outputs
