import argparse
import os
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from engines.cli import validate_config
import yaml
from torch.utils.data import DataLoader

from datasets.tri_modal_st_dataset import TriModalSTGraphDataset, collate_single_graph
from losses.graph_contrastive_loss import GraphContrastiveLoss
from losses.reconstruction_loss import ReconstructionLoss
from losses.tri_modal_consistency_loss import TriModalConsistencyLoss
from models.tri_modal_full_model import TrimodalSTGraphModel
from utils.seed import set_seed
from utils.train_utils import (
    AverageMeter,
    ensure_dir,
    load_yaml_config,
    move_batch_to_device,
    resolve_device,
    save_checkpoint,
)


TRI_MODAL_BRANCHES = ("omics1", "omics2", "omics3")
TRI_MODAL_BRANCH_MODE = "trimodal_fusion"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train trimodal ST graph SSL model")
    parser.add_argument("--config", type=str, default="configs/tri_modal_default.yaml")
    return parser.parse_args(argv)


def build_loader(
    data_cfg: Dict[str, Any],
    input_path: str,
    shuffle: bool,
) -> DataLoader:
    dataset = TriModalSTGraphDataset(
        input_path=input_path,
        file_pattern=data_cfg.get("file_pattern", "*.h5ad"),
        omics1_obsm_key=data_cfg["omics1_obsm_key"],
        omics2_obsm_key=data_cfg["omics2_obsm_key"],
        omics3_obsm_key=data_cfg["omics3_obsm_key"],
        omics1_target_obsm_key=data_cfg.get("omics1_target_obsm_key", ""),
        omics2_target_obsm_key=data_cfg.get("omics2_target_obsm_key", ""),
        omics3_target_obsm_key=data_cfg.get("omics3_target_obsm_key", ""),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        collate_fn=collate_single_graph,
    )


def infer_trimodal_dimensions_from_sample(
    config: Dict[str, Any],
    sample: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    cfg = deepcopy(config)
    model_cfg = cfg["model"]
    model_cfg["branch_mode"] = TRI_MODAL_BRANCH_MODE

    for branch in TRI_MODAL_BRANCHES:
        feat_key = f"{branch}_feat"
        target_key = f"{branch}_target"
        in_dim_key = f"{branch}_in_dim"
        target_dim_key = f"{branch}_target_dim"

        if model_cfg.get(in_dim_key) is None:
            model_cfg[in_dim_key] = int(sample[feat_key].shape[-1])

        if model_cfg.get(target_dim_key) is None:
            if target_key in sample:
                model_cfg[target_dim_key] = int(sample[target_key].shape[-1])
            else:
                model_cfg[target_dim_key] = int(sample[feat_key].shape[-1])

    return cfg


def _reconstruction_loss_for_branch(
    branch: str,
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    recon_loss_fn: ReconstructionLoss,
) -> torch.Tensor:
    target = batch.get(f"{branch}_target", batch[f"{branch}_feat"])
    recon = outputs[f"{branch}_recon"]
    if recon.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"{branch}_recon dim {recon.shape[-1]} != {branch}_target dim {target.shape[-1]}"
        )
    return recon_loss_fn(recon, target)


def compute_losses(
    cfg: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    recon_loss_fn: ReconstructionLoss,
    contrastive_loss_fn: GraphContrastiveLoss,
    consistency_loss_fn: Optional[TriModalConsistencyLoss],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    zero = outputs["z"].sum() * 0.0

    loss_omics1 = _reconstruction_loss_for_branch("omics1", batch, outputs, recon_loss_fn)
    loss_omics2 = _reconstruction_loss_for_branch("omics2", batch, outputs, recon_loss_fn)
    loss_omics3 = _reconstruction_loss_for_branch("omics3", batch, outputs, recon_loss_fn)

    z = outputs["z"]
    perm = torch.randperm(z.shape[0], device=z.device)
    z_corrupt = z[perm]
    loss_contrastive = contrastive_loss_fn(z, z_corrupt, outputs["edge_index"])

    loss_consistency = zero
    if float(cfg["loss"].get("lambda_consistency", 0.0)) > 0:
        if consistency_loss_fn is None:
            raise RuntimeError("consistency_loss_fn is required when lambda_consistency > 0")
        loss_consistency = consistency_loss_fn(
            outputs["omics1_multi_scale_embedding"],
            outputs["omics2_multi_scale_embedding"],
            outputs["omics3_multi_scale_embedding"],
            outputs["z"],
        )

    loss_cfg = cfg["loss"]
    total = (
        float(loss_cfg["lambda_omics1"]) * loss_omics1
        + float(loss_cfg["lambda_omics2"]) * loss_omics2
        + float(loss_cfg["lambda_omics3"]) * loss_omics3
        + float(loss_cfg["lambda_contrastive"]) * loss_contrastive
        + float(loss_cfg.get("lambda_consistency", 0.0)) * loss_consistency
    )

    loss_dict = {
        "loss_total": total,
        "loss_omics1": loss_omics1,
        "loss_omics2": loss_omics2,
        "loss_omics3": loss_omics3,
        "loss_contrastive": loss_contrastive,
        "loss_consistency": loss_consistency,
    }
    return total, loss_dict


def run_epoch(
    cfg: Dict[str, Any],
    model: TrimodalSTGraphModel,
    loader: DataLoader,
    device: torch.device,
    recon_loss_fn: ReconstructionLoss,
    contrastive_loss_fn: GraphContrastiveLoss,
    consistency_loss_fn: Optional[TriModalConsistencyLoss],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    train_mode: bool,
) -> Dict[str, float]:
    meters = {
        "loss_total": AverageMeter(),
        "loss_omics1": AverageMeter(),
        "loss_omics2": AverageMeter(),
        "loss_omics3": AverageMeter(),
        "loss_contrastive": AverageMeter(),
        "loss_consistency": AverageMeter(),
    }

    model.train(mode=train_mode)
    contrastive_loss_fn.train(mode=train_mode)
    if consistency_loss_fn is not None:
        consistency_loss_fn.train(mode=train_mode)

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    loss_params = list(contrastive_loss_fn.parameters())
    if consistency_loss_fn is not None:
        loss_params.extend(list(consistency_loss_fn.parameters()))
    clip_params = list(model.parameters()) + loss_params

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
            total, loss_dict = compute_losses(
                cfg,
                batch,
                outputs,
                recon_loss_fn,
                contrastive_loss_fn,
                consistency_loss_fn,
            )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(total).backward()
                grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                optimizer.step()

        for key, meter in meters.items():
            meter.update(float(loss_dict[key].detach().cpu().item()))

        if train_mode and step % int(cfg["train"].get("log_interval", 10)) == 0:
            print(
                f"step={step} "
                f"total={meters['loss_total'].avg:.4f} "
                f"omics1={meters['loss_omics1'].avg:.4f} "
                f"omics2={meters['loss_omics2'].avg:.4f} "
                f"omics3={meters['loss_omics3'].avg:.4f} "
                f"contrastive={meters['loss_contrastive'].avg:.4f} "
                f"consistency={meters['loss_consistency'].avg:.4f}"
            )

    return {key: meter.avg for key, meter in meters.items()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)
    validate_config(cfg, backend="full", action="train", trimodal=True)
    cfg.setdefault("model", {})["branch_mode"] = TRI_MODAL_BRANCH_MODE

    set_seed(int(cfg.get("seed", 42)))

    train_loader = build_loader(
        cfg["data"],
        input_path=cfg["data"]["train_dir"],
        shuffle=True,
    )
    first_sample = train_loader.dataset[0]
    cfg = infer_trimodal_dimensions_from_sample(cfg, first_sample)

    device = resolve_device(str(cfg["train"].get("device", "cuda")))
    model = TrimodalSTGraphModel(cfg).to(device)

    recon_loss_fn = ReconstructionLoss()
    contrastive_loss_fn = GraphContrastiveLoss(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        **cfg["loss"]["contrastive"],
    ).to(device)

    consistency_loss_fn: Optional[TriModalConsistencyLoss] = None
    if float(cfg["loss"].get("lambda_consistency", 0.0)) > 0:
        consistency_loss_fn = TriModalConsistencyLoss(
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            **cfg["loss"]["consistency"],
        ).to(device)

    optimizer_params = [model.parameters(), contrastive_loss_fn.parameters()]
    if consistency_loss_fn is not None:
        optimizer_params.append(consistency_loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        chain(*optimizer_params),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    val_loader = None
    val_dir = cfg["data"].get("val_dir", "")
    if isinstance(val_dir, str) and val_dir.strip():
        val_loader = build_loader(cfg["data"], input_path=val_dir, shuffle=False)

    output_dir = cfg["train"].get("output_dir", "./outputs/tri_modal")
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    best_metric = float("inf")
    epochs = int(cfg["train"].get("epochs", 30))

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            cfg,
            model,
            train_loader,
            device,
            recon_loss_fn,
            contrastive_loss_fn,
            consistency_loss_fn,
            optimizer,
            scaler,
            train_mode=True,
        )

        msg = (
            f"epoch={epoch}/{epochs} "
            f"train_total={train_metrics['loss_total']:.4f} "
            f"omics1={train_metrics['loss_omics1']:.4f} "
            f"omics2={train_metrics['loss_omics2']:.4f} "
            f"omics3={train_metrics['loss_omics3']:.4f} "
            f"contrastive={train_metrics['loss_contrastive']:.4f} "
            f"consistency={train_metrics['loss_consistency']:.4f}"
        )

        monitor = train_metrics["loss_total"]
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    cfg,
                    model,
                    val_loader,
                    device,
                    recon_loss_fn,
                    contrastive_loss_fn,
                    consistency_loss_fn,
                    optimizer,
                    scaler,
                    train_mode=False,
                )
            msg += (
                f" val_total={val_metrics['loss_total']:.4f}"
                f" val_consistency={val_metrics['loss_consistency']:.4f}"
            )
            monitor = val_metrics["loss_total"]

        print(msg)

        save_every = int(cfg["train"].get("save_every", 0))
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"),
                model,
                optimizer,
                epoch,
                cfg,
                contrastive_loss=contrastive_loss_fn,
                consistency_loss=consistency_loss_fn,
            )

        if monitor < best_metric:
            best_metric = monitor
            save_checkpoint(
                os.path.join(output_dir, "best.pt"),
                model,
                optimizer,
                epoch,
                cfg,
                contrastive_loss=contrastive_loss_fn,
                consistency_loss=consistency_loss_fn,
            )


if __name__ == "__main__":
    main()
