from typing import Any, Dict, List

import torch
import torch.nn as nn

from models.decoders import MLPDecoder
from models.encoders import FeatureAdapter, PointNeXtGraphBackbone
from models.fusion import ConcatMLP3Fusion
from models.gat_blocks import GATStack
from models.graph_builder import SpatialGraphBuilder
from models.layer_transformer import LayerwiseTransformerAggregator
from models.tri_modal_attention import TRI_MODAL_BRANCHES, SpatialTriModalAttention
from utils.graph_utils import neighbor_mean


VALID_GRAPH_BACKBONE_TYPES = {"gat", "pointnext"}


def _as_mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _resolve_encoder_cfg(model_cfg: Dict[str, Any], branch: str) -> Dict[str, Any]:
    encoder_cfg = _as_mapping(model_cfg.get(f"{branch}_encoder", {}), f"model.{branch}_encoder")
    feature_adapter_cfg = encoder_cfg.get("feature_adapter")
    if feature_adapter_cfg is not None:
        return _as_mapping(feature_adapter_cfg, f"model.{branch}_encoder.feature_adapter")
    return encoder_cfg


def _resolve_branch_graph_backbone_type(model_cfg: Dict[str, Any], branch: str) -> str:
    graph_backbone_cfg = _as_mapping(model_cfg.get("graph_backbone", {}), "model.graph_backbone")
    branch_cfg = _as_mapping(graph_backbone_cfg.get(branch, {}), f"model.graph_backbone.{branch}")

    backbone_type = str(branch_cfg.get("type", "gat"))
    if backbone_type not in VALID_GRAPH_BACKBONE_TYPES:
        raise ValueError(
            f"model.graph_backbone.{branch}.type must be one of: gat, pointnext"
        )
    return backbone_type


def _resolve_branch_use_layer_transformer(model_cfg: Dict[str, Any], branch: str) -> bool:
    graph_backbone_cfg = _as_mapping(model_cfg.get("graph_backbone", {}), "model.graph_backbone")
    branch_cfg = _as_mapping(graph_backbone_cfg.get(branch, {}), f"model.graph_backbone.{branch}")
    return bool(branch_cfg.get("use_layer_transformer", True))


def _resolve_branch_gat_num_layers(model_cfg: Dict[str, Any], branch: str) -> int:
    gat_cfg = _as_mapping(model_cfg.get("gat", {}), "model.gat")
    value = gat_cfg.get(f"{branch}_num_layers")
    if value is None:
        value = gat_cfg.get("num_layers", 0)
    num_layers = int(value)
    if num_layers < 0:
        raise ValueError(f"model.gat.{branch}_num_layers must be >= 0, got {num_layers}")
    return num_layers


class TrimodalSTGraphModel(nn.Module):
    """
    Three-modality ST graph model.

    This class is intentionally parallel to MultimodalSTGraphModel and does not
    change the existing two-modality training or inference entrypoints.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_dim"])

        graph_cfg = _as_mapping(model_cfg["graph"], "model.graph")
        graph_backbone_cfg = _as_mapping(model_cfg.get("graph_backbone", {}), "model.graph_backbone")
        pointnext_cfg = _as_mapping(
            graph_backbone_cfg.get("pointnext", {}),
            "model.graph_backbone.pointnext",
        )
        attn_cfg = _as_mapping(model_cfg["cross_modal_attention"], "model.cross_modal_attention")
        gat_cfg = _as_mapping(model_cfg.get("gat", {}), "model.gat")
        tr_cfg = _as_mapping(model_cfg["layer_transformer"], "model.layer_transformer")
        fusion_cfg = _as_mapping(model_cfg.get("fusion", {}), "model.fusion")
        omics1_aug_cfg = _as_mapping(
            model_cfg.get("omics1_augmentation", {}),
            "model.omics1_augmentation",
        )

        self.branch_mode = "trimodal_fusion"
        self.hidden_dim = hidden_dim
        self.omics1_augmentation_enabled = bool(omics1_aug_cfg.get("enabled", True))
        self.omics1_augmentation_alpha = float(omics1_aug_cfg.get("alpha", 0.3))

        self.adapter_dims: Dict[str, int] = {}
        self.backbone_types: Dict[str, str] = {}
        self.use_layer_transformer: Dict[str, bool] = {}
        self.num_gat_layers: Dict[str, int] = {}

        self.adapters = nn.ModuleDict()
        self.pointnext_backbones = nn.ModuleDict()
        self.gat_backbones = nn.ModuleDict()
        self.layer_aggs = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for branch in TRI_MODAL_BRANCHES:
            encoder_cfg = _resolve_encoder_cfg(model_cfg, branch)
            adapter_dim = encoder_cfg.get(
                "adapter_dim",
                encoder_cfg.get(f"{branch}_adapter_dim", hidden_dim),
            )
            adapter_dim_int = int(adapter_dim) if adapter_dim is not None else None
            self.adapter_dims[branch] = adapter_dim_int if adapter_dim_int is not None else hidden_dim
            self.adapters[branch] = FeatureAdapter(
                in_dim=int(model_cfg[f"{branch}_in_dim"]),
                hidden_dim=hidden_dim,
                adapter_type=str(encoder_cfg.get("adapter_type", "mlp_bottleneck")),
                adapter_dim=adapter_dim_int,
                dropout=float(encoder_cfg.get("dropout", 0.1)),
                force_linear=bool(encoder_cfg.get("force_linear", False)),
            )

            self.backbone_types[branch] = _resolve_branch_graph_backbone_type(model_cfg, branch)
            self.use_layer_transformer[branch] = _resolve_branch_use_layer_transformer(
                model_cfg,
                branch,
            )
            self.num_gat_layers[branch] = _resolve_branch_gat_num_layers(model_cfg, branch)

        self.graph_builder = SpatialGraphBuilder(
            top_k=int(graph_cfg.get("top_k", 6)),
            symmetrize=bool(graph_cfg.get("symmetrize", True)),
            add_self_loops=bool(graph_cfg.get("add_self_loops", False)),
            chunk_size=int(graph_cfg.get("chunk_size", 512)),
        )

        self.cross_attn = SpatialTriModalAttention(
            hidden_dim=hidden_dim,
            num_heads=int(attn_cfg["num_heads"]),
            dropout=float(attn_cfg["dropout"]),
        )

        pointnext_num_layers = int(pointnext_cfg.get("num_layers", 3))
        pointnext_expansion = int(pointnext_cfg.get("expansion", 4))
        max_neighbors = pointnext_cfg.get("max_neighbors")
        if max_neighbors is None:
            max_neighbors = int(graph_cfg.get("top_k", 6))

        for branch in TRI_MODAL_BRANCHES:
            if self.backbone_types[branch] == "pointnext":
                self.pointnext_backbones[branch] = PointNeXtGraphBackbone(
                    hidden_dim=hidden_dim,
                    num_layers=pointnext_num_layers,
                    expansion=pointnext_expansion,
                    dropout=float(pointnext_cfg.get("dropout", 0.1)),
                    max_neighbors=int(max_neighbors),
                    pooling=str(pointnext_cfg.get("pooling", "max")),
                )
                if self.use_layer_transformer[branch]:
                    self.layer_aggs[branch] = LayerwiseTransformerAggregator(
                        num_layers=pointnext_num_layers,
                        hidden_dim=hidden_dim,
                        num_heads=int(tr_cfg["num_heads"]),
                        ff_dim=int(tr_cfg["ff_dim"]),
                        dropout=float(tr_cfg["dropout"]),
                        pooling=str(tr_cfg["pooling"]),
                    )
            elif self.num_gat_layers[branch] > 0:
                self.gat_backbones[branch] = GATStack(
                    hidden_dim=hidden_dim,
                    num_layers=self.num_gat_layers[branch],
                    num_heads=int(gat_cfg["num_heads"]),
                    dropout=float(gat_cfg["dropout"]),
                    negative_slope=float(gat_cfg["leaky_relu_negative_slope"]),
                )
                if self.use_layer_transformer[branch] and self.num_gat_layers[branch] >= 2:
                    self.layer_aggs[branch] = LayerwiseTransformerAggregator(
                        num_layers=self.num_gat_layers[branch],
                        hidden_dim=hidden_dim,
                        num_heads=int(tr_cfg["num_heads"]),
                        ff_dim=int(tr_cfg["ff_dim"]),
                        dropout=float(tr_cfg["dropout"]),
                        pooling=str(tr_cfg["pooling"]),
                    )

        for branch in TRI_MODAL_BRANCHES:
            self.decoders[branch] = MLPDecoder(
                in_dim=hidden_dim,
                out_dim=int(model_cfg[f"{branch}_target_dim"]),
                hidden_dim=self.adapter_dims[branch],
            )

        mlp_hidden_dim = fusion_cfg.get(
            "mlp_hidden_dim",
            fusion_cfg.get("gate_hidden_dim", hidden_dim),
        )
        self.fusion = ConcatMLP3Fusion(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=int(mlp_hidden_dim),
            dropout=float(fusion_cfg.get("dropout", 0.1)),
        )

    def _apply_graph_backbone(
        self,
        branch: str,
        fused_feat: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        backbone_type = self.backbone_types[branch]

        if backbone_type == "pointnext":
            if branch not in self.pointnext_backbones:
                raise RuntimeError(f"{branch} PointNeXt backbone is not initialized")
            final_stage, stage_outputs = self.pointnext_backbones[branch](
                coords,
                fused_feat,
                edge_index,
                return_stage_outputs=True,
            )
            outputs[f"{branch}_pointnext_outputs"] = stage_outputs
            outputs[f"{branch}_backbone_outputs"] = stage_outputs
            if self.use_layer_transformer[branch] and len(stage_outputs) >= 2:
                if branch not in self.layer_aggs:
                    raise RuntimeError(f"{branch} layer transformer is not initialized")
                return self.layer_aggs[branch](stage_outputs)
            return final_stage

        outputs[f"{branch}_backbone_outputs"] = []
        if self.num_gat_layers[branch] == 0:
            return fused_feat

        if branch not in self.gat_backbones:
            raise RuntimeError(f"{branch} GAT backbone is not initialized")

        layer_outputs: List[torch.Tensor] = self.gat_backbones[branch](fused_feat, edge_index)
        outputs[f"{branch}_layer_outputs"] = layer_outputs
        outputs[f"{branch}_backbone_outputs"] = layer_outputs
        if self.use_layer_transformer[branch] and len(layer_outputs) >= 2:
            if branch not in self.layer_aggs:
                raise RuntimeError(f"{branch} layer transformer is not initialized")
            return self.layer_aggs[branch](layer_outputs)
        return layer_outputs[-1]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        coords = batch["coords"]
        features = {branch: batch[f"{branch}_feat"] for branch in TRI_MODAL_BRANCHES}

        edge_index = self.graph_builder.build(
            coords,
            cache_key=batch.get("graph_cache_key"),
        )

        if self.omics1_augmentation_enabled:
            neighbor_omics1_feat, _ = neighbor_mean(
                features["omics1"],
                edge_index,
                exclude_self=True,
            )
            omics1_feat_aug = (
                features["omics1"]
                + self.omics1_augmentation_alpha * neighbor_omics1_feat
            )
        else:
            omics1_feat_aug = features["omics1"]
        features["omics1"] = omics1_feat_aug

        embeddings = {
            branch: self.adapters[branch](features[branch])
            for branch in TRI_MODAL_BRANCHES
        }

        omics1_fused, omics2_fused, omics3_fused = self.cross_attn(
            omics1_feat=embeddings["omics1"],
            omics2_feat=embeddings["omics2"],
            omics3_feat=embeddings["omics3"],
            edge_index=edge_index,
            coords=coords,
        )
        fused = {
            "omics1": omics1_fused,
            "omics2": omics2_fused,
            "omics3": omics3_fused,
        }

        outputs: Dict[str, Any] = {
            "branch_mode": self.branch_mode,
            "omics1_feat_aug": omics1_feat_aug,
            "edge_index": edge_index,
        }
        for branch in TRI_MODAL_BRANCHES:
            outputs[f"{branch}_emb"] = embeddings[branch]
            outputs[f"{branch}_fused"] = fused[branch]

        multi_scale_embeddings: Dict[str, torch.Tensor] = {}
        for branch in TRI_MODAL_BRANCHES:
            multi_scale_embeddings[branch] = self._apply_graph_backbone(
                branch=branch,
                fused_feat=fused[branch],
                coords=coords,
                edge_index=edge_index,
                outputs=outputs,
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
