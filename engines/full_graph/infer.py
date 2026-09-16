import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from engines.cli import validate_config

from datasets.st_dataset import STGraphDataset, collate_single_graph
from models.full_model import MultimodalSTGraphModel
from utils.train_utils import (
    ensure_dir,
    infer_dimensions_from_sample,
    load_yaml_config,
    move_batch_to_device,
    resolve_branch_mode,
    resolve_device,
)

import scanpy as sc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for multimodal ST graph SSL model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--output_path_h5ad", type=str, default="./outputs/infer_results.h5ad")
    parser.add_argument(
        "--output_dir_h5ad",
        type=str,
        default="",
        help=(
            "Optional directory for per-sample h5ad outputs. When input_dir contains "
            "multiple h5ad files, each output is written as {sample_id}_fusion.h5ad."
        ),
    )
    parser.add_argument("--branch_mode", type=str, default="")
    parser.add_argument("--save_reconstruction", action="store_true")
    return parser.parse_args(argv)


def _write_outputs_to_h5ad(
    file_path: str,
    outputs: Dict[str, torch.Tensor],
    save_recon: bool,
    output_path_h5ad: str,
) -> None:
    adata = sc.read_h5ad(file_path)
    adata.obsm["z"] = outputs["z"].detach().cpu().numpy()

    if "fusion_gate" in outputs:
        adata.obsm["fusion_gate"] = outputs["fusion_gate"].detach().cpu().numpy()
    elif "fusion_gate" in adata.obsm:
        del adata.obsm["fusion_gate"]

    for key in ["omics1_recon", "omics2_recon"]:
        if save_recon and key in outputs:
            adata.obsm[key] = outputs[key].detach().cpu().numpy()
        elif key in adata.obsm:
            del adata.obsm[key]

    adata.write_h5ad(output_path_h5ad)


def _resolve_output_h5ad_path(
    *,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    sample_id: str,
    dataset_size: int,
) -> str:
    output_dir_h5ad = args.output_dir_h5ad or cfg.get("infer", {}).get("output_dir_h5ad", "")
    if output_dir_h5ad:
        ensure_dir(output_dir_h5ad)
        return str(Path(output_dir_h5ad) / f"{sample_id}_fusion.h5ad")

    if dataset_size > 1:
        raise ValueError(
            "Multiple inference inputs require --output_dir_h5ad or infer.output_dir_h5ad "
            "so per-sample h5ad outputs do not overwrite each other."
        )

    return args.output_path_h5ad


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)

    ckpt_path = args.checkpoint or cfg.get("infer", {}).get("checkpoint_path", "")
    if not ckpt_path:
        raise ValueError("Please provide checkpoint path via --checkpoint or infer.checkpoint_path")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "config" in ckpt:
        cfg = ckpt["config"]
    validate_config(cfg, backend="full", action="infer")
    if args.branch_mode:
        cfg.setdefault("model", {})["branch_mode"] = args.branch_mode
    branch_mode = resolve_branch_mode(cfg)

    input_dir = args.input_dir or cfg["data"].get("infer_dir", "")
    if not input_dir:
        raise ValueError("Please provide input_dir via --input_dir or data.infer_dir")

    dataset = STGraphDataset(
        input_path=input_dir,
        file_pattern=cfg["data"].get("file_pattern", "*.h5ad"),
        omics1_obsm_key=cfg["data"]["omics1_obsm_key"],
        omics2_obsm_key=cfg["data"]["omics2_obsm_key"],
        omics1_target_obsm_key=cfg["data"].get("omics1_target_obsm_key", ""),
        omics2_target_obsm_key=cfg["data"].get("omics2_target_obsm_key", ""),
    )
    first_sample = dataset[0]
    cfg = infer_dimensions_from_sample(cfg, first_sample)
    branch_mode = resolve_branch_mode(cfg)

    device = resolve_device(str(cfg["train"].get("device", "cuda")))

    model = MultimodalSTGraphModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    save_recon = bool(args.save_reconstruction or cfg["infer"].get("save_reconstruction", False))
    output_path = args.output_path or cfg["infer"].get("output_path", "./outputs/infer_embeddings.pt")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_dir(output_dir)

    results: List[Dict[str, Any]] = []
    for i in range(len(dataset)):
        sample = collate_single_graph([dataset[i]])
        sample_id = sample.get("sample_id", f"sample_{i}")
        file_path = dataset.files[i]
        batch = move_batch_to_device(sample, device)

        with torch.no_grad():
            outputs = model(batch)

        item: Dict[str, Any] = {
            "sample_id": sample_id,
            "branch_mode": branch_mode,
            "z": outputs["z"].detach().cpu(),
        }
        if save_recon and "omics1_recon" in outputs:
            item["omics1_recon"] = outputs["omics1_recon"].detach().cpu()
        if save_recon and "omics2_recon" in outputs:
            item["omics2_recon"] = outputs["omics2_recon"].detach().cpu()

        output_h5ad = _resolve_output_h5ad_path(
            args=args,
            cfg=cfg,
            sample_id=str(sample_id),
            dataset_size=len(dataset),
        )
        item["output_h5ad"] = output_h5ad

        results.append(item)
        _write_outputs_to_h5ad(file_path, outputs, save_recon, output_h5ad)
        print(f"Updated inference results in: {output_h5ad}")

    torch.save({"branch_mode": branch_mode, "results": results}, output_path)
    print(f"Saved inference outputs to: {output_path}")


if __name__ == "__main__":
    main()
