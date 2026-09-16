from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.tile_utils import load_json


MODE_TO_KEYS = {
    "rna_protein": ("rna", "protein"),
    "rna_he": ("rna", "path"),
    "rna_protein_he": ("rna", "protein", "path"),
}


class XeniumTileDataset(Dataset):
    def __init__(self, manifest_path: str | Path, mode: str) -> None:
        if mode not in MODE_TO_KEYS:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(MODE_TO_KEYS)}")
        self.manifest = load_json(manifest_path)
        self.processed_h5 = str(self.manifest.get("processed_h5", ""))
        self.tiles: List[dict] = list(self.manifest["tiles"])
        self.mode = mode
        self.modalities = MODE_TO_KEYS[mode]
        if "path" in self.modalities:
            processed_files = {
                self._processed_h5_for_record(record) for record in self.tiles
            }
            for processed_h5 in processed_files:
                with h5py.File(processed_h5, "r") as handle:
                    if "path_feat" not in handle:
                        raise KeyError(
                            f"{processed_h5} has no path_feat; run pathology feature extraction first"
                        )
                    completed = int(handle.attrs.get("path_feat_completed_rows", 0))
                    expected = int(handle["coords"].shape[0])
                    if completed < expected:
                        raise RuntimeError(
                            f"H&E features are incomplete in {processed_h5}: "
                            f"{completed}/{expected} rows. Resume pathology feature extraction."
                        )

    def _processed_h5_for_record(self, record: dict) -> str:
        processed_h5 = str(record.get("processed_h5", self.processed_h5))
        if not processed_h5:
            raise KeyError(
                "Manifest must define root-level 'processed_h5' or per-tile "
                "'processed_h5' entries"
            )
        return processed_h5

    def __len__(self) -> int:
        return len(self.tiles)

    @staticmethod
    def _read_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
        order = np.argsort(rows)
        sorted_rows = rows[order]
        values = np.asarray(dataset[sorted_rows])
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return values[inverse]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.tiles[index]
        processed_h5 = self._processed_h5_for_record(record)
        tile = np.load(record["path"])
        global_index = np.asarray(tile["global_node_index"], dtype=np.int64)
        sample: Dict[str, torch.Tensor | str] = {
            "sample_id": str(record["sample_id"]),
            "tile_id": str(record["tile_id"]),
            "global_node_index": torch.from_numpy(global_index),
            "edge_index": torch.from_numpy(
                np.asarray(tile["edge_index"], dtype=np.int64)
            ),
            "core_mask": torch.from_numpy(np.asarray(tile["core_mask"], dtype=bool)),
            "tile_center": torch.from_numpy(np.asarray(tile["center"], dtype=np.float32)),
        }
        with h5py.File(processed_h5, "r") as handle:
            sample["coords"] = torch.from_numpy(
                self._read_rows(handle["coords"], global_index).astype(np.float32)
            )
            for modality in self.modalities:
                feature_key = f"{modality}_feat"
                target_key = f"{modality}_target"
                if feature_key not in handle:
                    raise KeyError(
                        f"{self.processed_h5} does not contain dataset {feature_key!r}"
                    )
                sample[feature_key] = torch.from_numpy(
                    self._read_rows(handle[feature_key], global_index).astype(np.float32)
                )
                if target_key in handle:
                    sample[target_key] = torch.from_numpy(
                        self._read_rows(handle[target_key], global_index).astype(np.float32)
                    )

        if self.mode == "rna_protein":
            sample["omics1_feat"] = sample["rna_feat"]
            sample["omics2_feat"] = sample["protein_feat"]
            sample["omics1_target"] = sample["rna_feat"]
            sample["omics2_target"] = sample.get("protein_target", sample["protein_feat"])
        elif self.mode == "rna_he":
            sample["omics1_feat"] = sample["rna_feat"]
            sample["omics2_feat"] = sample["path_feat"]
            sample["omics1_target"] = sample["rna_feat"]
            sample["omics2_target"] = sample.get("path_target", sample["path_feat"])
        else:
            sample["omics1_feat"] = sample["rna_feat"]
            sample["omics2_feat"] = sample["protein_feat"]
            sample["omics3_feat"] = sample["path_feat"]
            sample["omics1_target"] = sample["rna_feat"]
            sample["omics2_target"] = sample.get("protein_target", sample["protein_feat"])
            sample["omics3_target"] = sample.get("path_target", sample["path_feat"])
        return sample


def collate_single_tile(
    batch: Sequence[Dict[str, torch.Tensor | str]],
) -> Dict[str, torch.Tensor | str]:
    if len(batch) != 1:
        raise ValueError("Xenium scalable training uses one variable-size tile per batch")
    return batch[0]
