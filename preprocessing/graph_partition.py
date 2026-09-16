from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Iterable, List, Sequence

import h5py
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from utils.tile_utils import ensure_dir, load_json, save_json


def build_spatial_graph(
    coords: np.ndarray,
    top_k: int = 6,
    symmetrize: bool = True,
) -> sparse.csr_matrix:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be [N,2], got {coords.shape}")
    num_nodes = coords.shape[0]
    if num_nodes == 0:
        return sparse.csr_matrix((0, 0), dtype=np.uint8)
    neighbors = min(int(top_k) + 1, num_nodes)
    model = NearestNeighbors(n_neighbors=neighbors, algorithm="kd_tree")
    indices = model.fit(coords).kneighbors(return_distance=False)
    rows = np.repeat(np.arange(num_nodes), indices.shape[1])
    cols = indices.reshape(-1)
    keep = rows != cols
    graph = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.uint8), (rows[keep], cols[keep])),
        shape=(num_nodes, num_nodes),
    )
    if symmetrize:
        graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def graph_hop_context(
    graph: sparse.csr_matrix,
    core: np.ndarray,
    num_hops: int,
) -> np.ndarray:
    visited = np.zeros(graph.shape[0], dtype=bool)
    visited[core] = True
    frontier = np.asarray(core, dtype=np.int64)
    for _ in range(int(num_hops)):
        if frontier.size == 0:
            break
        starts = graph.indptr[frontier]
        ends = graph.indptr[frontier + 1]
        neighbors = np.concatenate(
            [graph.indices[start:end] for start, end in zip(starts, ends)]
        )
        new_nodes = np.unique(neighbors[~visited[neighbors]])
        visited[new_nodes] = True
        frontier = new_nodes
    return np.flatnonzero(visited)


def _split_core(
    coords: np.ndarray,
    indices: np.ndarray,
    core_target: int,
    max_nodes: int,
    graph: sparse.csr_matrix,
    num_hops: int,
    seed: int,
) -> List[np.ndarray]:
    context = graph_hop_context(graph, indices, num_hops)
    if indices.size <= core_target and context.size <= max_nodes:
        return [indices]
    if indices.size <= 1:
        raise RuntimeError(
            f"Cannot fit a one-node core with {context.size} context nodes into max_nodes={max_nodes}"
        )
    cluster = MiniBatchKMeans(
        n_clusters=2,
        random_state=seed,
        batch_size=min(4096, indices.size),
        n_init=3,
    )
    labels = cluster.fit_predict(coords[indices])
    parts = []
    for label in range(2):
        child = indices[labels == label]
        if child.size == 0:
            midpoint = indices.size // 2
            ordered = indices[np.argsort(coords[indices, 0], kind="stable")]
            children = [ordered[:midpoint], ordered[midpoint:]]
            break
    else:
        children = [indices[labels == 0], indices[labels == 1]]
    for offset, child in enumerate(children):
        parts.extend(
            _split_core(
                coords,
                child,
                core_target,
                max_nodes,
                graph,
                num_hops,
                seed + offset + 1,
            )
        )
    return parts


def partition_core_nodes(
    coords: np.ndarray,
    graph: sparse.csr_matrix,
    core_target: int = 2000,
    max_nodes: int = 5000,
    num_hops: int = 3,
    seed: int = 42,
) -> List[np.ndarray]:
    num_nodes = coords.shape[0]
    n_clusters = max(1, int(np.ceil(num_nodes / core_target)))
    if n_clusters == 1:
        initial = [np.arange(num_nodes, dtype=np.int64)]
    else:
        labels = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(8192, num_nodes),
            n_init=3,
        ).fit_predict(coords)
        initial = [
            np.flatnonzero(labels == label).astype(np.int64)
            for label in range(n_clusters)
            if np.any(labels == label)
        ]
    cores: List[np.ndarray] = []
    for index, core in enumerate(initial):
        cores.extend(
            _split_core(
                coords,
                core,
                core_target,
                max_nodes,
                graph,
                num_hops,
                seed + 1000 * index,
            )
        )
    coverage = np.zeros(num_nodes, dtype=np.int16)
    for core in cores:
        coverage[core] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Core partition does not cover every node exactly once")
    return cores


def induced_edge_index(
    graph: sparse.csr_matrix,
    global_nodes: np.ndarray,
) -> np.ndarray:
    subgraph = graph[global_nodes][:, global_nodes].tocoo()
    return np.vstack([subgraph.row, subgraph.col]).astype(np.int64)


def create_tiles(
    processed_h5: str | Path,
    graph_path: str | Path,
    tile_dir: str | Path,
    sample_id: str,
    top_k: int = 6,
    core_target: int = 2000,
    max_nodes: int = 5000,
    num_hops: int = 3,
    seed: int = 42,
) -> dict:
    with h5py.File(processed_h5, "r") as handle:
        coords = np.asarray(handle["coords"], dtype=np.float32)
    graph = build_spatial_graph(coords, top_k=top_k, symmetrize=True)
    ensure_dir(Path(graph_path).parent)
    sparse.save_npz(graph_path, graph, compressed=True)
    cores = partition_core_nodes(
        coords,
        graph,
        core_target=core_target,
        max_nodes=max_nodes,
        num_hops=num_hops,
        seed=seed,
    )
    output_dir = ensure_dir(tile_dir)
    records = []
    node_coverage = np.zeros(coords.shape[0], dtype=np.int32)
    context_coverage = np.zeros(coords.shape[0], dtype=np.int32)
    for tile_index, core in enumerate(cores):
        global_nodes = graph_hop_context(graph, core, num_hops)
        if global_nodes.size > max_nodes:
            raise RuntimeError(
                f"Tile {tile_index} has {global_nodes.size} nodes after recursive splitting"
            )
        core_mask = np.isin(global_nodes, core, assume_unique=True)
        edge_index = induced_edge_index(graph, global_nodes)
        node_coverage[global_nodes] += 1
        context_coverage[global_nodes[~core_mask]] += 1
        tile_id = f"{sample_id}_tile_{tile_index:05d}"
        path = output_dir / f"{tile_id}.npz"
        np.savez_compressed(
            path,
            global_node_index=global_nodes.astype(np.int64),
            edge_index=edge_index,
            core_mask=core_mask,
            center=coords[core].mean(axis=0).astype(np.float32),
        )
        records.append(
            {
                "sample_id": sample_id,
                "tile_id": tile_id,
                "path": str(path.resolve()),
                "num_nodes": int(global_nodes.size),
                "num_core": int(core.size),
                "num_edges": int(edge_index.shape[1]),
            }
        )
    manifest = {
        "sample_id": sample_id,
        "processed_h5": str(Path(processed_h5).resolve()),
        "graph_path": str(Path(graph_path).resolve()),
        "top_k": int(top_k),
        "core_target": int(core_target),
        "max_nodes": int(max_nodes),
        "num_hops": int(num_hops),
        "num_cells": int(coords.shape[0]),
        "num_tiles": len(records),
        "coverage_min": int(node_coverage.min()),
        "coverage_max": int(node_coverage.max()),
        "context_coverage_mean": float(context_coverage.mean()),
        "tiles": records,
    }
    save_json(output_dir / "manifest.json", manifest)
    summary_path = Path(processed_h5).with_suffix(".summary.json")
    if summary_path.is_file():
        summary = load_json(summary_path)
        summary["graph"] = {
            "graph_path": str(Path(graph_path).resolve()),
            "top_k": int(top_k),
            "core_target": int(core_target),
            "max_nodes": int(max_nodes),
            "num_hops": int(num_hops),
            "num_tiles": len(records),
            "coverage_min": int(node_coverage.min()),
            "coverage_max": int(node_coverage.max()),
            "context_coverage_mean": float(context_coverage.mean()),
            "tile_nodes_min": int(min(record["num_nodes"] for record in records)),
            "tile_nodes_max": int(max(record["num_nodes"] for record in records)),
        }
        save_json(summary_path, summary)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spatial graph tiles from HDF5 coordinates")
    parser.add_argument("--processed-h5", required=True)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--core-target", type=int, default=2000)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--num-hops", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_tiles(
        processed_h5=args.processed_h5,
        graph_path=args.graph_path,
        tile_dir=args.tile_dir,
        sample_id=args.sample_id,
        top_k=args.top_k,
        core_target=args.core_target,
        max_nodes=args.max_nodes,
        num_hops=args.num_hops,
        seed=args.seed,
    )
    print(f"created {manifest['num_tiles']} tiles for {manifest['num_cells']} cells")


if __name__ == "__main__":
    main()
