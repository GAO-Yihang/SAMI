from __future__ import annotations

import argparse
import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import yaml
from losses.cross_modal_consistency_loss import CrossModalConsistencyLoss
from losses.tri_modal_consistency_loss import TriModalConsistencyLoss
from torch.utils.data import DataLoader

from utils.tile_utils import ensure_dir, load_config, resolve_device, set_seed
from engines.cli import validate_config
from datasets.tile_dataset import XeniumTileDataset, collate_single_tile
from losses.tile_losses import (
    MaskedGraphContrastiveLoss,
    compute_dual_losses,
    compute_trimodal_losses,
)
from models.tile_model import ScalableFMRLModel, ScalableTriModalFMRLModel


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def infer_dimensions(config: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config["model"]
    mode = config["data"]["mode"]
    branches = ("omics1", "omics2", "omics3") if mode == "rna_protein_he" else (
        "omics1",
        "omics2",
    )
    for branch in branches:
        if model_cfg.get(f"{branch}_in_dim") is None:
            model_cfg[f"{branch}_in_dim"] = int(sample[f"{branch}_feat"].shape[1])
        if model_cfg.get(f"{branch}_target_dim") is None:
            target = sample.get(f"{branch}_target", sample[f"{branch}_feat"])
            model_cfg[f"{branch}_target_dim"] = int(target.shape[1])
    return config


def create_components(config: Dict[str, Any], device: torch.device):
    mode = config["data"]["mode"]
    hidden_dim = int(config["model"]["hidden_dim"])
    if mode == "rna_protein_he":
        model = ScalableTriModalFMRLModel(config).to(device)
        consistency = TriModalConsistencyLoss(
            hidden_dim=hidden_dim,
            **config["loss"]["consistency"],
        ).to(device)
    else:
        model = ScalableFMRLModel(config).to(device)
        consistency = CrossModalConsistencyLoss(
            hidden_dim=hidden_dim,
            **config["loss"]["consistency"],
        ).to(device)
    contrastive = MaskedGraphContrastiveLoss(
        hidden_dim=hidden_dim,
        **config["loss"]["contrastive"],
    ).to(device)
    return model, contrastive, consistency


def save_checkpoint(
    path: Path,
    epoch: int,
    config: Dict[str, Any],
    model: torch.nn.Module,
    contrastive: torch.nn.Module,
    consistency: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "config": config,
            "model_state": model.state_dict(),
            "contrastive_loss_state": contrastive.state_dict(),
            "consistency_loss_state": consistency.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def train_epoch(
    config: Dict[str, Any],
    model: torch.nn.Module,
    contrastive: MaskedGraphContrastiveLoss,
    consistency: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    contrastive.train()
    consistency.train()
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    accumulation = int(config["train"].get("gradient_accumulation", 1))
    if accumulation <= 0:
        raise ValueError("train.gradient_accumulation must be positive")
    optimizer.zero_grad(set_to_none=True)
    totals: Dict[str, float] = {}
    steps = 0
    parameters = list(
        chain(model.parameters(), contrastive.parameters(), consistency.parameters())
    )
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
            if config["data"]["mode"] == "rna_protein_he":
                total, metrics = compute_trimodal_losses(
                    config, batch, outputs, contrastive, consistency
                )
            else:
                total, metrics = compute_dual_losses(
                    config, batch, outputs, contrastive, consistency
                )
            scaled_total = total / accumulation
        scaler.scale(scaled_total).backward()
        should_step = step % accumulation == 0 or step == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            grad_clip = float(config["train"].get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        steps += 1
        if step % int(config["train"].get("log_interval", 10)) == 0:
            print(
                f"step={step}/{len(loader)} "
                + " ".join(f"{key}={value / steps:.4f}" for key, value in totals.items()),
                flush=True,
            )
    return {key: value / max(steps, 1) for key, value in totals.items()}


def run(config_path: str | Path, *, require_trimodal: bool = False) -> None:
    config = load_config(config_path)
    validate_config(config, backend="tiled", action="train", trimodal=require_trimodal)
    set_seed(int(config.get("seed", 42)))
    dataset = XeniumTileDataset(
        config["data"]["tile_manifest"],
        mode=config["data"]["mode"],
    )
    first_sample = dataset[0]
    config = infer_dimensions(config, first_sample)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=bool(config["data"].get("shuffle", True)),
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=bool(config["data"].get("pin_memory", False)),
        collate_fn=collate_single_tile,
    )
    device = resolve_device(str(config["train"].get("device", "cuda")))
    model, contrastive, consistency = create_components(config, device)
    optimizer = torch.optim.AdamW(
        chain(model.parameters(), contrastive.parameters(), consistency.parameters()),
        lr=float(config["train"].get("lr", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(
        config["train"].get("amp", True)
    ))
    output_dir = ensure_dir(config["train"]["output_dir"])
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    best = float("inf")
    stale_epochs = 0
    patience = int(config["train"].get("early_stopping_patience", 10))
    for epoch in range(1, int(config["train"].get("epochs", 50)) + 1):
        metrics = train_epoch(
            config,
            model,
            contrastive,
            consistency,
            loader,
            optimizer,
            scaler,
            device,
        )
        print(
            f"epoch={epoch} "
            + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()),
            flush=True,
        )
        save_checkpoint(
            output_dir / "last.pt",
            epoch,
            config,
            model,
            contrastive,
            consistency,
            optimizer,
        )
        monitor = metrics["loss_total"]
        if monitor < best:
            best = monitor
            stale_epochs = 0
            save_checkpoint(
                output_dir / "best.pt",
                epoch,
                config,
                model,
                contrastive,
                consistency,
                optimizer,
            )
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAMI on spatial tiles")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, require_trimodal: bool = False) -> None:
    args = parse_args(argv)
    run(args.config, require_trimodal=require_trimodal)


if __name__ == "__main__":
    main()
