import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import yaml


VALID_BRANCH_MODES = {"omics1", "omics2", "multimodal_fusion"}
VALID_GRAPH_BACKBONE_TYPES = {"gat", "pointnext"}


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    return batch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n


def resolve_branch_mode(config: Dict[str, Any]) -> str:
    model_cfg = config["model"]
    branch_mode = str(model_cfg.get("branch_mode", "multimodal_fusion"))
    if branch_mode not in VALID_BRANCH_MODES:
        raise ValueError(
            "model.branch_mode must be one of: omics1, omics2, multimodal_fusion"
        )
    return branch_mode


def uses_omics1_branch(branch_mode: str) -> bool:
    if branch_mode not in VALID_BRANCH_MODES:
        raise ValueError(f"Unsupported branch_mode: {branch_mode}")
    return branch_mode in {"omics1", "multimodal_fusion"}


def uses_omics2_branch(branch_mode: str) -> bool:
    if branch_mode not in VALID_BRANCH_MODES:
        raise ValueError(f"Unsupported branch_mode: {branch_mode}")
    return branch_mode in {"omics2", "multimodal_fusion"}


def resolve_omics2_encoder_cfg(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    omics2_encoder_cfg = model_cfg.get("omics2_encoder", {})
    if not isinstance(omics2_encoder_cfg, dict):
        raise TypeError("model.omics2_encoder must be a mapping")
    return dict(omics2_encoder_cfg)


def resolve_branch_graph_backbone_type(model_cfg: Dict[str, Any], branch: str) -> str:
    if branch not in {"omics1", "omics2"}:
        raise ValueError(f"branch must be 'omics1' or 'omics2', got {branch!r}")

    graph_backbone_cfg = model_cfg.setdefault("graph_backbone", {})
    branch_cfg = graph_backbone_cfg.setdefault(branch, {})
    if not isinstance(branch_cfg, dict):
        raise TypeError(f"model.graph_backbone.{branch} must be a mapping")

    backbone_type = str(branch_cfg.get("type", "gat"))
    if backbone_type not in VALID_GRAPH_BACKBONE_TYPES:
        raise ValueError(
            f"model.graph_backbone.{branch}.type must be one of: gat, pointnext"
        )
    return backbone_type


def resolve_branch_use_layer_transformer(model_cfg: Dict[str, Any], branch: str) -> bool:
    if branch not in {"omics1", "omics2"}:
        raise ValueError(f"branch must be 'omics1' or 'omics2', got {branch!r}")

    graph_backbone_cfg = model_cfg.setdefault("graph_backbone", {})
    branch_cfg = graph_backbone_cfg.setdefault(branch, {})
    if not isinstance(branch_cfg, dict):
        raise TypeError(f"model.graph_backbone.{branch} must be a mapping")

    return bool(branch_cfg.get("use_layer_transformer", True))


def resolve_branch_gat_num_layers(model_cfg: Dict[str, Any], branch: str) -> int:
    if branch not in {"omics1", "omics2"}:
        raise ValueError(f"branch must be 'omics1' or 'omics2', got {branch!r}")

    gat_cfg = model_cfg.setdefault("gat", {})
    value = gat_cfg.get(f"{branch}_num_layers")
    if value is None:
        value = gat_cfg.get("num_layers", 0)
    num_layers = int(value)
    if num_layers < 0:
        raise ValueError(f"model.gat.{branch}_num_layers must be >= 0, got {num_layers}")
    return num_layers


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Dict[str, Any],
    contrastive_loss: Optional[torch.nn.Module] = None,
    consistency_loss: Optional[torch.nn.Module] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }
    if contrastive_loss is not None:
        payload["contrastive_loss_state"] = contrastive_loss.state_dict()
    if consistency_loss is not None:
        payload["consistency_loss_state"] = consistency_loss.state_dict()
    torch.save(payload, path)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def infer_dimensions_from_sample(
    config: Dict[str, Any],
    sample: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    cfg = deepcopy(config)
    model_cfg = cfg["model"]
    branch_mode = resolve_branch_mode(cfg)

    if model_cfg.get("omics1_in_dim") is None:
        model_cfg["omics1_in_dim"] = int(sample["omics1_feat"].shape[-1])

    omics2_in = int(sample["omics2_feat"].shape[-1])
    if model_cfg.get("omics2_in_dim") is None:
        model_cfg["omics2_in_dim"] = omics2_in

    if uses_omics1_branch(branch_mode) and model_cfg.get("omics1_target_dim") is None:
        if "omics1_target" in sample:
            model_cfg["omics1_target_dim"] = int(sample["omics1_target"].shape[-1])
        else:
            model_cfg["omics1_target_dim"] = int(sample["omics1_feat"].shape[-1])

    if uses_omics2_branch(branch_mode) and model_cfg.get("omics2_target_dim") is None:
        if "omics2_target" in sample:
            model_cfg["omics2_target_dim"] = int(sample["omics2_target"].shape[-1])
        else:
            model_cfg["omics2_target_dim"] = omics2_in

    return cfg
