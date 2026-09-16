from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader

from utils.tile_utils import ensure_dir, resolve_device
from datasets.tile_dataset import XeniumTileDataset, collate_single_tile
from models.tile_model import ScalableFMRLModel, ScalableTriModalFMRLModel
from engines.tiled.train import move_batch
from engines.cli import validate_config, validate_single_sample_manifest


def read_sparse_csr(group: h5py.Group) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            np.asarray(group["data"]),
            np.asarray(group["indices"]),
            np.asarray(group["indptr"]),
        ),
        shape=tuple(group.attrs["shape"]),
    )


def occurrence_weights(
    coords: torch.Tensor,
    center: torch.Tensor,
    core_mask: torch.Tensor,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(coords - center.unsqueeze(0), dim=1)
    core_distance = distance[core_mask]
    radius = torch.quantile(core_distance, 0.9).clamp_min(1e-6)
    context_weight = 0.5 * torch.exp(-torch.square(distance / radius))
    return torch.where(core_mask, torch.ones_like(context_weight), context_weight)


def build_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    if config["data"]["mode"] == "rna_protein_he":
        return ScalableTriModalFMRLModel(config).to(device)
    return ScalableFMRLModel(config).to(device)


def run(
    checkpoint_path: str | Path,
    output_path: str | Path,
    device_name: str = "cuda",
    *,
    require_trimodal: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError("Tiled inference requires a checkpoint containing its config")
    config = checkpoint["config"]
    validate_config(config, backend="tiled", action="infer", trimodal=require_trimodal)
    validate_single_sample_manifest(config["data"]["tile_manifest"])
    device = resolve_device(device_name)
    dataset = XeniumTileDataset(
        config["data"]["tile_manifest"],
        config["data"]["mode"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collate_single_tile,
    )
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    num_cells = int(dataset.manifest["num_cells"])
    hidden_dim = int(config["model"]["hidden_dim"])
    keys = ["z", "omics1_multi_scale_embedding", "omics2_multi_scale_embedding"]
    output_names = {
        "z": "z",
        "omics1_multi_scale_embedding": "rna_embedding",
        "omics2_multi_scale_embedding": (
            "protein_embedding"
            if config["data"]["mode"] == "rna_protein"
            else "path_embedding"
        ),
    }
    if config["data"]["mode"] == "rna_protein_he":
        keys.append("omics3_multi_scale_embedding")
        output_names["omics2_multi_scale_embedding"] = "protein_embedding"
        output_names["omics3_multi_scale_embedding"] = "path_embedding"
    sums = {key: np.zeros((num_cells, hidden_dim), dtype=np.float64) for key in keys}
    squares = np.zeros((num_cells, hidden_dim), dtype=np.float64)
    weight_sum = np.zeros(num_cells, dtype=np.float64)
    coverage = np.zeros(num_cells, dtype=np.int32)
    context_coverage = np.zeros(num_cells, dtype=np.int32)

    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            outputs = model(batch)
            global_index = batch["global_node_index"].cpu().numpy()
            core_mask = batch["core_mask"]
            weights = occurrence_weights(
                batch["coords"], batch["tile_center"], core_mask
            ).cpu().numpy()
            for key in keys:
                values = outputs[key].float().cpu().numpy()
                np.add.at(sums[key], global_index, values * weights[:, None])
            z_values = outputs["z"].float().cpu().numpy()
            np.add.at(squares, global_index, np.square(z_values) * weights[:, None])
            np.add.at(weight_sum, global_index, weights)
            np.add.at(coverage, global_index, 1)
            np.add.at(context_coverage, global_index, (~core_mask).cpu().numpy().astype(np.int32))
            if step % 20 == 0:
                print(f"inference tiles {step}/{len(loader)}", flush=True)

    if np.any(weight_sum <= 0):
        raise RuntimeError("Some cells received no tile embedding")
    embeddings = {
        output_names[key]: (value / weight_sum[:, None]).astype(np.float32)
        for key, value in sums.items()
    }
    variance = np.maximum(
        squares / weight_sum[:, None] - np.square(embeddings["z"]),
        0,
    ).mean(axis=1)

    processed_h5 = dataset.processed_h5
    with h5py.File(processed_h5, "r") as handle:
        cell_ids = np.asarray(handle["cell_id"]).astype(str)
        coords = np.asarray(handle["coords"], dtype=np.float32)
        counts = read_sparse_csr(handle["raw_rna_csr"])
        gene_key = (
            "raw_rna_feature_names"
            if "raw_rna_feature_names" in handle
            else "rna_feature_names"
        )
        genes = np.asarray(handle[gene_key]).astype(str)
        obs_data = {
            "x_array": coords[:, 0],
            "y_array": coords[:, 1],
            "tile_coverage": coverage,
            "context_coverage": context_coverage,
            "embedding_variance": variance.astype(np.float32),
        }
        for key in ("total_rna_counts", "detected_genes", "path_valid_fraction"):
            if key in handle:
                obs_data[key] = np.asarray(handle[key])
    obs = pd.DataFrame(obs_data, index=pd.Index(cell_ids, name="cell_id"))
    var_names = genes
    if counts.shape[1] != len(var_names):
        var_names = np.asarray([f"gene_{index}" for index in range(counts.shape[1])])
    result = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(var_names.astype(str), name="feature")),
    )
    result.obsm["spatial"] = coords
    for key, value in embeddings.items():
        result.obsm[key] = value
    result.uns["resolved_config"] = config
    result.uns["tiling"] = {
        key: dataset.manifest[key]
        for key in (
            "top_k",
            "core_target",
            "max_nodes",
            "num_hops",
            "num_tiles",
        )
    }
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    result.write_h5ad(output_path, compression="gzip")
    with output_path.with_suffix(".qc.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_cells": num_cells,
                "coverage_min": int(coverage.min()),
                "coverage_max": int(coverage.max()),
                "embedding_variance_mean": float(variance.mean()),
                "embedding_variance_p99": float(np.quantile(variance, 0.99)),
            },
            handle,
            indent=2,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer and stitch SAMI tile embeddings")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, require_trimodal: bool = False) -> None:
    args = parse_args(argv)
    run(args.checkpoint, args.output, args.device, require_trimodal=require_trimodal)


if __name__ == "__main__":
    main()
