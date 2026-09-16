"""Route public commands without merging the two execution engines."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def dispatch(
    action: str,
    argv: Sequence[str] | None = None,
    *,
    trimodal: bool = False,
) -> None:
    if action not in {"train", "infer"}:
        raise ValueError(f"Unknown command: {action}")
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--backend",
        choices=("full", "tiled"),
        default="full",
        help="execution engine (default: full); select tiled explicitly for tile data",
    )
    routing, remaining = parser.parse_known_args(arguments)
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
        print(
            f"Parameters below are for --backend {routing.backend}. "
            "Use --backend full --help or --backend tiled --help for each engine.\n"
        )
        if trimodal and routing.backend == "tiled":
            print("Trimodal tiled commands require data.mode=rna_protein_he.\n")
    command = f"tri_{action}" if trimodal and routing.backend == "full" else action
    package = "full_graph" if routing.backend == "full" else "tiled"
    engine = importlib.import_module(f"engines.{package}.{command}")
    if routing.backend == "tiled":
        engine.main(remaining, require_trimodal=trimodal)
    else:
        engine.main(remaining)


def validate_config(
    config: Any,
    *,
    backend: str,
    action: str,
    trimodal: bool = False,
) -> None:
    """Validate the effective engine config, after checkpoint precedence is applied."""
    prefix = f"Invalid --backend {backend} {action} configuration"
    if not isinstance(config, Mapping):
        raise ValueError(f"{prefix}: expected a mapping")
    required_sections = ["data", "model"]
    if action == "train" or backend == "full":
        required_sections.append("train")
    if action == "train":
        required_sections.append("loss")
    for section in required_sections:
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"{prefix}: missing mapping '{section}'")
    data = config["data"]
    required = ["model.hidden_dim"]
    if backend == "full":
        if "tile_manifest" in data:
            raise ValueError(
                f"{prefix}: data.tile_manifest belongs to --backend tiled; "
                "full expects H5AD inputs and data.omics*_obsm_key"
            )
        required.extend(["data.omics1_obsm_key", "data.omics2_obsm_key"])
        if trimodal:
            required.append("data.omics3_obsm_key")
        if action == "train":
            required.append("data.train_dir")
    else:
        required.extend(["data.mode", "data.tile_manifest"])
        mode = data.get("mode")
        if mode and mode not in {"rna_protein", "rna_he", "rna_protein_he"}:
            raise ValueError(
                f"{prefix}: data.mode must be rna_protein, rna_he or rna_protein_he"
            )
        if trimodal and mode != "rna_protein_he":
            raise ValueError(
                f"{prefix}: tri_* with --backend tiled requires "
                "data.mode=rna_protein_he"
            )
        if action == "train":
            required.append("train.output_dir")
    missing = []
    for field in required:
        section, key = field.split(".")
        value = config[section].get(key)
        if value is None or value == "":
            missing.append(field)
    if missing:
        raise ValueError(f"{prefix}: missing {', '.join(missing)}")


def validate_single_sample_manifest(manifest_path: str | Path) -> None:
    """Stitching uses sample-local node indices and one processed H5 file."""
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("Tiled inference manifest must be a mapping")
    processed_h5 = manifest.get("processed_h5")
    records = manifest.get("tiles", [])
    sample_ids = {record.get("sample_id") for record in records}
    processed_files = {
        str(Path(record.get("processed_h5", processed_h5) or "").resolve())
        for record in records
    }
    is_joint = (
        int(manifest.get("num_samples", 1)) > 1
        or len(sample_ids) > 1
        or not processed_h5
        or "num_cells" not in manifest
        or (processed_files and processed_files != {str(Path(processed_h5).resolve())})
    )
    if is_joint:
        raise ValueError(
            "Tiled inference requires a single-sample manifest with root-level "
            "processed_h5 and num_cells; joint training manifests cannot be stitched. "
            "Use the checkpoint configured for the requested sample "
            "(for CRC_VisiumHD, the sample directory's best.pt)."
        )
    if not records:
        raise ValueError("Tiled inference manifest contains no tiles")
