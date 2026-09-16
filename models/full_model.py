from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from models.cross_modal_attention import SpatialCrossModalAttention
from models.decoders import MLPDecoder
from models.encoders import FeatureAdapter, PointNeXtGraphBackbone
from models.fusion import ConcatMLPFusion
from models.gat_blocks import GATStack
from models.graph_builder import SpatialGraphBuilder
from models.layer_transformer import LayerwiseTransformerAggregator
from utils.train_utils import (
    resolve_branch_gat_num_layers,
    resolve_branch_graph_backbone_type,
    resolve_branch_mode,
    resolve_branch_use_layer_transformer,
    resolve_omics2_encoder_cfg,
    uses_omics1_branch,
    uses_omics2_branch,
)
from utils.graph_utils import neighbor_mean


class MultimodalSTGraphModel(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_dim"])
        omics1_in_dim = int(model_cfg["omics1_in_dim"])
        omics1_encoder_cfg = model_cfg["omics1_encoder"]
        omics1_adapter_cfg = omics1_encoder_cfg.get("feature_adapter", {})
        omics2_encoder_cfg = resolve_omics2_encoder_cfg(model_cfg)
        omics1_aug_cfg = model_cfg.get("omics1_augmentation", {})
        graph_cfg = model_cfg["graph"]
        graph_backbone_cfg = model_cfg.setdefault("graph_backbone", {})
        pointnext_cfg = graph_backbone_cfg.setdefault("pointnext", {})

        self.branch_mode = resolve_branch_mode(config)
        self.use_omics1_branch = uses_omics1_branch(self.branch_mode)
        self.use_omics2_branch = uses_omics2_branch(self.branch_mode)
        self.omics1_backbone_type = resolve_branch_graph_backbone_type(model_cfg, "omics1")
        self.omics2_backbone_type = resolve_branch_graph_backbone_type(model_cfg, "omics2")
        self.omics1_use_layer_transformer = resolve_branch_use_layer_transformer(
            model_cfg,
            "omics1",
        )
        self.omics2_use_layer_transformer = resolve_branch_use_layer_transformer(
            model_cfg,
            "omics2",
        )
        self.omics1_num_gat_layers = resolve_branch_gat_num_layers(model_cfg, "omics1")
        self.omics2_num_gat_layers = resolve_branch_gat_num_layers(model_cfg, "omics2")
        self.omics1_augmentation_enabled = bool(omics1_aug_cfg.get("enabled", True))
        self.omics1_augmentation_alpha = float(omics1_aug_cfg.get("alpha", 0.3))

        omics1_adapter_dim = omics1_adapter_cfg.get("adapter_dim")
        self.omics1_adapter = FeatureAdapter(
            in_dim=omics1_in_dim,
            hidden_dim=hidden_dim,
            adapter_type=str(omics1_adapter_cfg.get("adapter_type", "mlp_bottleneck")),
            adapter_dim=int(omics1_adapter_dim) if omics1_adapter_dim is not None else None,
            dropout=float(omics1_adapter_cfg.get("dropout", 0.1)),
            force_linear=bool(omics1_adapter_cfg.get("force_linear", False)),
        )

        omics2_adapter_dim = omics2_encoder_cfg.get(
            "adapter_dim",
            omics2_encoder_cfg.get("omics2_adapter_dim", hidden_dim),
        )
        self.omics2_adapter = FeatureAdapter(
            in_dim=int(model_cfg["omics2_in_dim"]),
            hidden_dim=hidden_dim,
            adapter_type=str(omics2_encoder_cfg.get("adapter_type", "mlp_bottleneck")),
            adapter_dim=int(omics2_adapter_dim) if omics2_adapter_dim is not None else None,
            dropout=float(omics2_encoder_cfg.get("dropout", 0.1)),
            force_linear=bool(omics2_encoder_cfg.get("force_linear", False)),
        )

        self.graph_builder = SpatialGraphBuilder(
            top_k=int(graph_cfg.get("top_k", 6)),
            symmetrize=bool(graph_cfg.get("symmetrize", True)),
            add_self_loops=bool(graph_cfg.get("add_self_loops", False)),
            chunk_size=int(graph_cfg.get("chunk_size", 512)),
        )

        attn_cfg = model_cfg["cross_modal_attention"]
        self.cross_attn = SpatialCrossModalAttention(
            hidden_dim=hidden_dim,
            num_heads=int(attn_cfg["num_heads"]),
            dropout=float(attn_cfg["dropout"]),
        )

        gat_cfg = model_cfg["gat"]
        tr_cfg = model_cfg["layer_transformer"]
        pointnext_expansion = int(pointnext_cfg.get("expansion", 4))
        max_neighbors = pointnext_cfg.get("max_neighbors")
        if max_neighbors is None:
            max_neighbors = int(graph_cfg.get("top_k", 6))

        self.omics1_pointnext: Optional[PointNeXtGraphBackbone] = None
        self.omics1_gat: Optional[GATStack] = None
        self.omics1_agg: Optional[LayerwiseTransformerAggregator] = None
        self.omics2_pointnext: Optional[PointNeXtGraphBackbone] = None
        self.omics2_gat: Optional[GATStack] = None
        self.omics2_agg: Optional[LayerwiseTransformerAggregator] = None
        self.fusion: Optional[ConcatMLPFusion]
        self.omics1_decoder: Optional[MLPDecoder]
        self.omics2_decoder: Optional[MLPDecoder]

        if self.use_omics1_branch:
            if self.omics1_backbone_type == "pointnext":
                self.omics1_pointnext = PointNeXtGraphBackbone(
                    hidden_dim=hidden_dim,
                    num_layers=3,
                    expansion=pointnext_expansion,
                    dropout=float(pointnext_cfg.get("dropout", 0.1)),
                    max_neighbors=int(max_neighbors),
                    pooling=str(pointnext_cfg.get("pooling", "max")),
                )
                if self.omics1_use_layer_transformer:
                    self.omics1_agg = LayerwiseTransformerAggregator(
                        num_layers=3,
                        hidden_dim=hidden_dim,
                        num_heads=int(tr_cfg["num_heads"]),
                        ff_dim=int(tr_cfg["ff_dim"]),
                        dropout=float(tr_cfg["dropout"]),
                        pooling=str(tr_cfg["pooling"]),
                    )
            elif self.omics1_num_gat_layers > 0:
                self.omics1_gat = GATStack(
                    hidden_dim=hidden_dim,
                    num_layers=self.omics1_num_gat_layers,
                    num_heads=int(gat_cfg["num_heads"]),
                    dropout=float(gat_cfg["dropout"]),
                    negative_slope=float(gat_cfg["leaky_relu_negative_slope"]),
                )
                if self.omics1_use_layer_transformer and self.omics1_num_gat_layers >= 2:
                    self.omics1_agg = LayerwiseTransformerAggregator(
                        num_layers=self.omics1_num_gat_layers,
                        hidden_dim=hidden_dim,
                        num_heads=int(tr_cfg["num_heads"]),
                        ff_dim=int(tr_cfg["ff_dim"]),
                        dropout=float(tr_cfg["dropout"]),
                        pooling=str(tr_cfg["pooling"]),
                    )

        if self.use_omics2_branch:
            if self.omics2_backbone_type == "pointnext":
                self.omics2_pointnext = PointNeXtGraphBackbone(
                    hidden_dim=hidden_dim,
                    num_layers=3,
                    expansion=pointnext_expansion,
                    dropout=float(pointnext_cfg.get("dropout", 0.1)),
                    max_neighbors=int(max_neighbors),
                    pooling=str(pointnext_cfg.get("pooling", "max")),
                )
                if self.omics2_use_layer_transformer:
                    self.omics2_agg = LayerwiseTransformerAggregator(
                        num_layers=3,
                        hidden_dim=hidden_dim,
                        num_heads=int(tr_cfg["num_heads"]),
                        ff_dim=int(tr_cfg["ff_dim"]),
                        dropout=float(tr_cfg["dropout"]),
                        pooling=str(tr_cfg["pooling"]),
                    )
            elif self.omics2_num_gat_layers > 0:
                self.omics2_gat = GATStack(
                    hidden_dim=hidden_dim,
                    num_layers=self.omics2_num_gat_layers,
                    num_heads=int(gat_cfg["num_heads"]),
                    dropout=float(gat_cfg["dropout"]),
                    negative_slope=float(gat_cfg["leaky_relu_negative_slope"]),
                )
                if self.omics2_use_layer_transformer and self.omics2_num_gat_layers >= 2:
                    self.omics2_agg = LayerwiseTransformerAggregator(
                        num_layers=self.omics2_num_gat_layers,
                        hidden_dim=hidden_dim,
                        num_heads=int(tr_cfg["num_heads"]),
                        ff_dim=int(tr_cfg["ff_dim"]),
                        dropout=float(tr_cfg["dropout"]),
                        pooling=str(tr_cfg["pooling"]),
                    )

        if self.use_omics1_branch:
            self.omics1_decoder = MLPDecoder(
                in_dim=hidden_dim,
                out_dim=int(model_cfg["omics1_target_dim"]),
                hidden_dim=int(omics1_adapter_dim)
                if omics1_adapter_dim is not None
                else hidden_dim,
            )
        else:
            self.omics1_decoder = None

        if self.use_omics2_branch:
            self.omics2_decoder = MLPDecoder(
                in_dim=hidden_dim,
                out_dim=int(model_cfg["omics2_target_dim"]),
                hidden_dim=int(omics2_adapter_dim)
                if omics2_adapter_dim is not None
                else hidden_dim,
            )
        else:
            self.omics2_decoder = None

        if self.branch_mode == "multimodal_fusion":
            fusion_cfg = model_cfg["fusion"]
            mlp_hidden_dim = fusion_cfg.get(
                "mlp_hidden_dim",
                fusion_cfg.get("gate_hidden_dim", hidden_dim),
            )
            self.fusion = ConcatMLPFusion(
                hidden_dim=hidden_dim,
                mlp_hidden_dim=int(mlp_hidden_dim),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
            )
        else:
            self.fusion = None

    def _apply_graph_backbone(
        self,
        branch: str,
        fused_feat: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        outputs: Dict[str, Any],
    ) -> torch.Tensor:
        if branch == "omics1":
            backbone_type = self.omics1_backbone_type
            pointnext = self.omics1_pointnext
            gat = self.omics1_gat
            agg = self.omics1_agg
            num_gat_layers = self.omics1_num_gat_layers
            use_layer_transformer = self.omics1_use_layer_transformer
        else:
            backbone_type = self.omics2_backbone_type
            pointnext = self.omics2_pointnext
            gat = self.omics2_gat
            agg = self.omics2_agg
            num_gat_layers = self.omics2_num_gat_layers
            use_layer_transformer = self.omics2_use_layer_transformer

        if backbone_type == "pointnext":
            if pointnext is None:
                raise RuntimeError(f"{branch} PointNeXt backbone is not initialized")
            final_stage, stage_outputs = pointnext(
                coords,
                fused_feat,
                edge_index,
                return_stage_outputs=True,
            )
            outputs[f"{branch}_pointnext_outputs"] = stage_outputs
            outputs[f"{branch}_backbone_outputs"] = stage_outputs
            if use_layer_transformer and len(stage_outputs) >= 2:
                if agg is None:
                    raise RuntimeError(f"{branch} layer transformer is not initialized")
                return agg(stage_outputs)
            return final_stage

        outputs[f"{branch}_backbone_outputs"] = []
        if num_gat_layers == 0:
            return fused_feat

        if gat is None:
            raise RuntimeError(f"{branch} GAT backbone is not initialized")

        layer_outputs: List[torch.Tensor] = gat(fused_feat, edge_index)
        outputs[f"{branch}_layer_outputs"] = layer_outputs
        outputs[f"{branch}_backbone_outputs"] = layer_outputs
        if use_layer_transformer and len(layer_outputs) >= 2:
            if agg is None:
                raise RuntimeError(f"{branch} layer transformer is not initialized")
            return agg(layer_outputs)
        return layer_outputs[-1]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        coords = batch["coords"]
        omics1_feat = batch["omics1_feat"]
        omics2_feat = batch["omics2_feat"]

        edge_index = self.graph_builder.build(
            coords,
            cache_key=batch.get("graph_cache_key"),
        )

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
        }

        omics1_multi_scale_embedding: Optional[torch.Tensor] = None
        omics2_multi_scale_embedding: Optional[torch.Tensor] = None

        if self.use_omics1_branch:
            omics1_multi_scale_embedding = self._apply_graph_backbone(
                branch="omics1",
                fused_feat=omics1_fused,
                coords=coords,
                edge_index=edge_index,
                outputs=outputs,
            )
            outputs["omics1_multi_scale_embedding"] = omics1_multi_scale_embedding

        if self.use_omics2_branch:
            omics2_multi_scale_embedding = self._apply_graph_backbone(
                branch="omics2",
                fused_feat=omics2_fused,
                coords=coords,
                edge_index=edge_index,
                outputs=outputs,
            )
            outputs["omics2_multi_scale_embedding"] = omics2_multi_scale_embedding

        if self.branch_mode == "multimodal_fusion":
            if (
                omics1_multi_scale_embedding is None
                or omics2_multi_scale_embedding is None
                or self.fusion is None
                or self.omics1_decoder is None
                or self.omics2_decoder is None
            ):
                raise RuntimeError("fusion mode requires both branches, fusion, and decoders")
            z = self.fusion(omics1_multi_scale_embedding, omics2_multi_scale_embedding)
            outputs["z"] = z
            outputs["omics1_recon"] = self.omics1_decoder(z)
            outputs["omics2_recon"] = self.omics2_decoder(z)
        elif self.branch_mode == "omics1":
            if omics1_multi_scale_embedding is None or self.omics1_decoder is None:
                raise RuntimeError("omics1 mode requires the omics1 branch and decoder")
            outputs["z"] = omics1_multi_scale_embedding
            outputs["omics1_recon"] = self.omics1_decoder(omics1_multi_scale_embedding)
        elif self.branch_mode == "omics2":
            if omics2_multi_scale_embedding is None or self.omics2_decoder is None:
                raise RuntimeError("omics2 mode requires the omics2 branch and decoder")
            outputs["z"] = omics2_multi_scale_embedding
            outputs["omics2_recon"] = self.omics2_decoder(omics2_multi_scale_embedding)
        else:
            raise ValueError(f"Unsupported branch_mode: {self.branch_mode}")

        return outputs
