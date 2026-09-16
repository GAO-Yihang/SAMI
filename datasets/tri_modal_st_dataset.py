import glob
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import scanpy as sc


def _to_numpy_array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    to_array = getattr(value, "toarray", None)
    if callable(to_array):
        return np.asarray(to_array())

    return np.asarray(value)


class TriModalSTGraphDataset(Dataset):
    """
    Dataset for pre-extracted three-modality spot-level features in .h5ad.

    Required fields:
    - obs['x_array'], obs['y_array'] -> coords [N, 2]
    - obsm[omics1_obsm_key] -> omics1_feat [N, D1]
    - obsm[omics2_obsm_key] -> omics2_feat [N, D2]
    - obsm[omics3_obsm_key] -> omics3_feat [N, D3]
    """

    def __init__(
        self,
        input_path: str,
        file_pattern: str = "*.h5ad",
        omics1_obsm_key: str = "",
        omics2_obsm_key: str = "",
        omics3_obsm_key: str = "",
        omics1_target_obsm_key: Optional[str] = None,
        omics2_target_obsm_key: Optional[str] = None,
        omics3_target_obsm_key: Optional[str] = None,
    ) -> None:
        if not omics1_obsm_key:
            raise ValueError("omics1_obsm_key must be provided")
        if not omics2_obsm_key:
            raise ValueError("omics2_obsm_key must be provided")
        if not omics3_obsm_key:
            raise ValueError("omics3_obsm_key must be provided")

        self.input_path = input_path
        self.file_pattern = file_pattern
        self.omics1_obsm_key = omics1_obsm_key
        self.omics2_obsm_key = omics2_obsm_key
        self.omics3_obsm_key = omics3_obsm_key
        self.omics1_target_obsm_key = omics1_target_obsm_key or ""
        self.omics2_target_obsm_key = omics2_target_obsm_key or ""
        self.omics3_target_obsm_key = omics3_target_obsm_key or ""

        if os.path.isfile(input_path):
            self.files: List[str] = [input_path]
        else:
            pattern = os.path.join(input_path, "**", file_pattern)
            self.files = sorted(glob.glob(pattern, recursive=True))

        if not self.files:
            raise FileNotFoundError(
                f"No .h5ad files found under {input_path} with pattern {file_pattern}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def _extract_obsm(self, adata: "sc.AnnData", key: str, file_path: str) -> torch.Tensor:
        if key not in adata.obsm:
            available = list(adata.obsm.keys())
            raise KeyError(
                f"{file_path} missing obsm['{key}']. Available obsm keys: {available}"
            )

        arr = _to_numpy_array(adata.obsm[key])
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{file_path} obsm['{key}'] must be 2D, got shape {arr.shape}")

        return torch.from_numpy(np.asarray(arr, dtype=np.float32))

    def _validate(self, sample: Dict[str, torch.Tensor], file_path: str) -> None:
        required = ["coords", "omics1_feat", "omics2_feat", "omics3_feat"]
        for key in required:
            if key not in sample:
                raise KeyError(f"{file_path} missing required key: {key}")

        coords = sample["coords"]
        features = {
            "omics1_feat": sample["omics1_feat"],
            "omics2_feat": sample["omics2_feat"],
            "omics3_feat": sample["omics3_feat"],
        }

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"{file_path} coords must be [N,2], got {tuple(coords.shape)}")

        n = coords.shape[0]
        for key, tensor in features.items():
            if tensor.ndim != 2:
                raise ValueError(f"{file_path} {key} must be [N,D], got {tuple(tensor.shape)}")
            if tensor.shape[0] != n:
                raise ValueError(
                    f"{file_path} {key} has mismatched N {tensor.shape[0]} vs coords {n}"
                )

        for key in ["omics1_target", "omics2_target", "omics3_target"]:
            if key not in sample:
                continue
            tensor = sample[key]
            if tensor.ndim != 2:
                raise ValueError(f"{file_path} {key} must be [N,D], got {tuple(tensor.shape)}")
            if tensor.shape[0] != n:
                raise ValueError(
                    f"{file_path} {key} has mismatched N {tensor.shape[0]} vs coords {n}"
                )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_path = self.files[index]
        adata = sc.read_h5ad(file_path)

        for obs_key in ("x_array", "y_array"):
            if obs_key not in adata.obs.columns:
                raise KeyError(
                    f"{file_path} missing obs['{obs_key}']. "
                    f"Available obs columns: {list(adata.obs.columns)}"
                )

        x_coord = np.asarray(adata.obs["x_array"].to_numpy(), dtype=np.float32)
        y_coord = np.asarray(adata.obs["y_array"].to_numpy(), dtype=np.float32)
        coords = torch.from_numpy(np.stack([x_coord, y_coord], axis=1))

        out: Dict[str, torch.Tensor] = {
            "coords": coords,
            "omics1_feat": self._extract_obsm(adata, self.omics1_obsm_key, file_path),
            "omics2_feat": self._extract_obsm(adata, self.omics2_obsm_key, file_path),
            "omics3_feat": self._extract_obsm(adata, self.omics3_obsm_key, file_path),
        }

        if self.omics1_target_obsm_key:
            out["omics1_target"] = self._extract_obsm(
                adata,
                self.omics1_target_obsm_key,
                file_path,
            )
        if self.omics2_target_obsm_key:
            out["omics2_target"] = self._extract_obsm(
                adata,
                self.omics2_target_obsm_key,
                file_path,
            )
        if self.omics3_target_obsm_key:
            out["omics3_target"] = self._extract_obsm(
                adata,
                self.omics3_target_obsm_key,
                file_path,
            )

        self._validate(out, file_path)
        out["sample_id"] = os.path.splitext(os.path.basename(file_path))[0]
        out["graph_cache_key"] = os.path.abspath(file_path)
        return out


def collate_single_graph(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError(
            "This baseline uses one graph per batch. "
            "Set data.batch_size=1 or implement a custom variable-graph collate function."
        )
    return batch[0]
