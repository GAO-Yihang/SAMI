import argparse
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import torch

from engines.cli import validate_config

from datasets.tri_modal_st_dataset import TriModalSTGraphDataset, collate_single_graph
from models.tri_modal_full_model import TrimodalSTGraphModel
from utils.train_utils import (
    ensure_dir,
    load_yaml_config,
    move_batch_to_device,
    resolve_device,
)

import scanpy as sc


TRI_MODAL_BRANCHES = ("omics1", "omics2", "omics3")
TRI_MODAL_BRANCH_MODE = "trimodal_fusion"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for trimodal ST graph SSL model")
    parser.add_argument("--config", type=str, default="configs/tri_modal_default.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--output_path_h5ad", type=str, default="")
    parser.add_argument("--save_reconstruction", action="store_true")
    return parser.parse_args(argv)


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


def _write_outputs_to_h5ad(
    file_path: str,
    outputs: Dict[str, torch.Tensor],
    save_recon: bool,
    output_path_h5ad: str,
) -> None:
    output_dir = os.path.dirname(output_path_h5ad)
    if output_dir:
        ensure_dir(output_dir)

    adata = sc.read_h5ad(file_path)
    adata.obsm["z"] = outputs["z"].detach().cpu().numpy()

    for key in ["omics1_recon", "omics2_recon", "omics3_recon"]:
        if save_recon and key in outputs:
            adata.obsm[key] = outputs[key].detach().cpu().numpy()
        elif key in adata.obsm:
            del adata.obsm[key]

    adata.write_h5ad(output_path_h5ad)


def _resolve_h5ad_output_path(
    output_path_h5ad: str,
    file_path: str,
    sample_id: str,
    num_files: int,
) -> str:
    if num_files <= 1:
        return output_path_h5ad

    if output_path_h5ad.endswith(".h5ad"):
        root, ext = os.path.splitext(output_path_h5ad)
        return f"{root}_{sample_id}{ext}"

    ensure_dir(output_path_h5ad)
    return os.path.join(output_path_h5ad, os.path.basename(file_path))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)

    infer_cfg = cfg.get("infer", {})
    ckpt_path = args.checkpoint or infer_cfg.get("checkpoint_path", "")
    if not ckpt_path:
        raise ValueError("Please provide checkpoint path via --checkpoint or infer.checkpoint_path")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "config" in ckpt:
        cfg = ckpt["config"]
        infer_cfg = cfg.get("infer", {})
    validate_config(cfg, backend="full", action="infer", trimodal=True)
    cfg.setdefault("model", {})["branch_mode"] = TRI_MODAL_BRANCH_MODE

    input_dir = args.input_dir or cfg["data"].get("infer_dir", "")
    if not input_dir:
        raise ValueError("Please provide input_dir via --input_dir or data.infer_dir")

    dataset = TriModalSTGraphDataset(
        input_path=input_dir,
        file_pattern=cfg["data"].get("file_pattern", "*.h5ad"),
        omics1_obsm_key=cfg["data"]["omics1_obsm_key"],
        omics2_obsm_key=cfg["data"]["omics2_obsm_key"],
        omics3_obsm_key=cfg["data"]["omics3_obsm_key"],
        omics1_target_obsm_key=cfg["data"].get("omics1_target_obsm_key", ""),
        omics2_target_obsm_key=cfg["data"].get("omics2_target_obsm_key", ""),
        omics3_target_obsm_key=cfg["data"].get("omics3_target_obsm_key", ""),
    )
    first_sample = dataset[0]
    cfg = infer_trimodal_dimensions_from_sample(cfg, first_sample)

    device = resolve_device(str(cfg["train"].get("device", "cuda")))
    model = TrimodalSTGraphModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    save_recon = bool(args.save_reconstruction or infer_cfg.get("save_reconstruction", False))
    output_path = args.output_path or infer_cfg.get("output_path", "./outputs/tri_infer_embeddings.pt")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_dir(output_dir)

    output_path_h5ad = args.output_path_h5ad or infer_cfg.get(
        "output_path_h5ad",
        "./outputs/tri_infer_results.h5ad",
    )

    results: List[Dict[str, Any]] = []
    for i in range(len(dataset)):
        sample = collate_single_graph([dataset[i]])
        sample_id = str(sample.get("sample_id", f"sample_{i}"))
        file_path = dataset.files[i]
        batch = move_batch_to_device(sample, device)

        with torch.no_grad():
            outputs = model(batch)

        item: Dict[str, Any] = {
            "sample_id": sample_id,
            "branch_mode": TRI_MODAL_BRANCH_MODE,
            "z": outputs["z"].detach().cpu(),
        }
        if save_recon:
            for key in ["omics1_recon", "omics2_recon", "omics3_recon"]:
                if key in outputs:
                    item[key] = outputs[key].detach().cpu()

        results.append(item)

        h5ad_path = _resolve_h5ad_output_path(
            output_path_h5ad,
            file_path,
            sample_id,
            num_files=len(dataset),
        )
        _write_outputs_to_h5ad(file_path, outputs, save_recon, h5ad_path)
        print(f"Updated inference results in: {h5ad_path}")

    torch.save({"branch_mode": TRI_MODAL_BRANCH_MODE, "results": results}, output_path)
    print(f"Saved inference outputs to: {output_path}")


if __name__ == "__main__":
    main()
