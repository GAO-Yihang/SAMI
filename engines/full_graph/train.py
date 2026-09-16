import argparse
import os
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from engines.cli import validate_config
import yaml
from torch.utils.data import DataLoader

from datasets.st_dataset import STGraphDataset, collate_single_graph
from losses.cross_modal_consistency_loss import CrossModalConsistencyLoss
from losses.graph_contrastive_loss import GraphContrastiveLoss
from losses.reconstruction_loss import ReconstructionLoss
from models.full_model import MultimodalSTGraphModel
from utils.seed import set_seed
from utils.train_utils import (
    AverageMeter,
    ensure_dir,
    infer_dimensions_from_sample,
    load_yaml_config,
    move_batch_to_device,
    resolve_branch_mode,
    resolve_device,
    save_checkpoint,
    uses_omics1_branch,
    uses_omics2_branch,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal ST graph SSL model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--branch_mode", type=str, default="")
    return parser.parse_args(argv)


def build_loader(
    data_cfg: Dict[str, Any],
    input_path: str,
    shuffle: bool,
) -> DataLoader:
    dataset = STGraphDataset(
        input_path=input_path,
        file_pattern=data_cfg.get("file_pattern", "*.h5ad"),
        omics1_obsm_key=data_cfg["omics1_obsm_key"],
        omics2_obsm_key=data_cfg["omics2_obsm_key"],
        omics1_target_obsm_key=data_cfg.get("omics1_target_obsm_key", ""),
        omics2_target_obsm_key=data_cfg.get("omics2_target_obsm_key", ""),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        collate_fn=collate_single_graph,
    )
    return loader


def compute_losses(
    cfg: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    recon_loss_fn: ReconstructionLoss,
    contrastive_loss_fn: GraphContrastiveLoss,
    consistency_loss_fn: Optional[CrossModalConsistencyLoss],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    branch_mode = resolve_branch_mode(cfg)
    zero = outputs["z"].sum() * 0.0

    loss_omics1 = zero
    loss_omics2 = zero
    loss_consistency = zero

    if uses_omics1_branch(branch_mode):
        omics1_target = batch.get("omics1_target", batch["omics1_feat"])
        if outputs["omics1_recon"].shape[-1] != omics1_target.shape[-1]:
            raise ValueError(
                f"omics1_recon dim {outputs['omics1_recon'].shape[-1]} "
                f"!= omics1_target dim {omics1_target.shape[-1]}"
            )
        loss_omics1 = recon_loss_fn(outputs["omics1_recon"], omics1_target)

    if uses_omics2_branch(branch_mode):
        omics2_target = batch.get("omics2_target", batch["omics2_feat"])
        if outputs["omics2_recon"].shape[-1] != omics2_target.shape[-1]:
            raise ValueError(
                f"omics2_recon dim {outputs['omics2_recon'].shape[-1]} "
                f"!= omics2_target dim {omics2_target.shape[-1]}"
            )
        loss_omics2 = recon_loss_fn(outputs["omics2_recon"], omics2_target)

    z = outputs["z"]
    perm = torch.randperm(z.shape[0], device=z.device)
    z_corrupt = z[perm]
    loss_contrastive = contrastive_loss_fn(
        z,
        z_corrupt,
        outputs["edge_index"],
    )

    if branch_mode == "multimodal_fusion":
        if consistency_loss_fn is None:
            raise RuntimeError("consistency_loss_fn is required for multimodal_fusion mode")
        loss_consistency = consistency_loss_fn(
            outputs["omics1_multi_scale_embedding"],
            outputs["omics2_multi_scale_embedding"],
            outputs["z"],
        )

    loss_cfg = cfg["loss"]
    total = (
        float(loss_cfg["lambda_omics1"]) * loss_omics1
        + float(loss_cfg["lambda_omics2"]) * loss_omics2
        + float(loss_cfg["lambda_contrastive"]) * loss_contrastive
        + float(loss_cfg["lambda_consistency"]) * loss_consistency
    )

    loss_dict = {
        "loss_total": total,
        "loss_omics1": loss_omics1,
        "loss_omics2": loss_omics2,
        "loss_contrastive": loss_contrastive,
        "loss_consistency": loss_consistency,
    }
    return total, loss_dict


def run_epoch(
    cfg: Dict[str, Any],
    model: MultimodalSTGraphModel,
    loader: DataLoader,
    device: torch.device,
    recon_loss_fn: ReconstructionLoss,
    contrastive_loss_fn: GraphContrastiveLoss,
    consistency_loss_fn: Optional[CrossModalConsistencyLoss],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    train_mode: bool,
) -> Dict[str, float]:
    meters = {
        "loss_total": AverageMeter(),
        "loss_omics1": AverageMeter(),
        "loss_omics2": AverageMeter(),
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

        for k, meter in meters.items():
            meter.update(float(loss_dict[k].detach().cpu().item()))

        if train_mode and step % int(cfg["train"].get("log_interval", 10)) == 0:
            print(
                f"step={step} "
                f"total={meters['loss_total'].avg:.4f} "
                f"omics1={meters['loss_omics1'].avg:.4f} "
                f"omics2={meters['loss_omics2'].avg:.4f} "
                f"contrastive={meters['loss_contrastive'].avg:.4f} "
                f"consistency={meters['loss_consistency'].avg:.4f}"
            )

    return {k: v.avg for k, v in meters.items()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)
    validate_config(cfg, backend="full", action="train")
    if args.branch_mode:
        cfg.setdefault("model", {})["branch_mode"] = args.branch_mode

    set_seed(int(cfg.get("seed", 42)))

    train_loader = build_loader(
        cfg["data"],
        input_path=cfg["data"]["train_dir"],
        shuffle=True,
    )
    first_sample = train_loader.dataset[0]
    cfg = infer_dimensions_from_sample(cfg, first_sample)
    branch_mode = resolve_branch_mode(cfg)

    device = resolve_device(str(cfg["train"].get("device", "cuda")))

    model = MultimodalSTGraphModel(cfg).to(device)
    recon_loss_fn = ReconstructionLoss()
    contrastive_loss_fn = GraphContrastiveLoss(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        **cfg["loss"]["contrastive"],
    ).to(device)
    consistency_loss_fn: Optional[CrossModalConsistencyLoss] = None
    if branch_mode == "multimodal_fusion":
        consistency_loss_fn = CrossModalConsistencyLoss(
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
        val_loader = build_loader(
            cfg["data"],
            input_path=val_dir,
            shuffle=False,
        )

    output_dir = cfg["train"].get("output_dir", "./outputs")
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

        # save_every = int(cfg["train"].get("save_every", 1))
        # if epoch % save_every == 0:
        #     ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
        #     save_checkpoint(
        #         ckpt_path,
        #         model,
        #         optimizer,
        #         epoch,
        #         cfg,
        #         contrastive_loss=contrastive_loss_fn,
        #         consistency_loss=consistency_loss_fn,
        #     )

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
