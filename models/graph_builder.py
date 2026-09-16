from typing import Dict, Optional

import torch


class SpatialGraphBuilder:
    def __init__(
        self,
        top_k: int = 6,
        symmetrize: bool = True,
        add_self_loops: bool = False,
        chunk_size: int = 512,
    ) -> None:
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.top_k = int(top_k)
        self.symmetrize = bool(symmetrize)
        self.add_self_loops = bool(add_self_loops)
        self.chunk_size = int(chunk_size)
        self._graph_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _normalize_cache_key(cache_key: Optional[str]) -> Optional[str]:
        if cache_key is None:
            return None
        key = str(cache_key).strip()
        return key or None

    @staticmethod
    def _squared_euclidean_distance_chunk(
        coords: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        a = coords[start:end]
        b = coords
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1, keepdim=True).transpose(0, 1)
        ab = a @ b.transpose(0, 1)
        return (a2 + b2 - 2.0 * ab).clamp_min(0.0)

    @staticmethod
    def _coalesce_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if edge_index.numel() == 0:
            return edge_index

        pairs = set()
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            if 0 <= s < num_nodes and 0 <= d < num_nodes:
                pairs.add((s, d))

        if not pairs:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        sorted_pairs = sorted(pairs)
        coalesced_src = torch.tensor(
            [p[0] for p in sorted_pairs],
            dtype=torch.long,
            device=edge_index.device,
        )
        coalesced_dst = torch.tensor(
            [p[1] for p in sorted_pairs],
            dtype=torch.long,
            device=edge_index.device,
        )
        return torch.stack([coalesced_src, coalesced_dst], dim=0)

    @torch.no_grad()
    def _compute_graph(self, coords_cpu: torch.Tensor) -> torch.Tensor:
        num_nodes = coords_cpu.shape[0]
        edge_device = coords_cpu.device

        if num_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long, device=edge_device)

        rows = []
        cols = []
        k = min(self.top_k, max(num_nodes - 1, 0))

        if k > 0:
            for start in range(0, num_nodes, self.chunk_size):
                end = min(start + self.chunk_size, num_nodes)
                dist2 = self._squared_euclidean_distance_chunk(coords_cpu, start, end)
                row_idx = torch.arange(end - start, device=edge_device)
                src_idx = torch.arange(start, end, device=edge_device)
                dist2[row_idx, src_idx] = float("inf")

                top_dist, top_idx = torch.topk(dist2, k=k, dim=1, largest=False)
                src = src_idx.unsqueeze(1).expand(-1, k).reshape(-1)
                dst = top_idx.reshape(-1)
                valid = torch.isfinite(top_dist.reshape(-1)) & (src != dst)
                rows.append(src[valid])
                cols.append(dst[valid])

        if rows:
            edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_device)

        if self.symmetrize and edge_index.numel() > 0:
            rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, rev], dim=1)
            edge_index = self._coalesce_edges(edge_index, num_nodes)

        if self.add_self_loops:
            loop_idx = torch.arange(num_nodes, device=edge_device)
            loop_edges = torch.stack([loop_idx, loop_idx], dim=0)
            edge_index = torch.cat([edge_index, loop_edges], dim=1)
            edge_index = self._coalesce_edges(edge_index, num_nodes)

        return edge_index

    @torch.no_grad()
    def build(
        self,
        coords: torch.Tensor,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            coords: [N, 2] spatial coordinates.
        Returns:
            edge_index: [2, E] sparse top-k distance graph.
        """
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be [N, 2], got {tuple(coords.shape)}")

        cache_key_norm = self._normalize_cache_key(cache_key)
        if cache_key_norm is not None and cache_key_norm in self._graph_cache:
            return self._graph_cache[cache_key_norm].to(device=coords.device)

        coords_cpu = coords.detach().to(device="cpu", dtype=torch.float32)
        edge_index_cpu = self._compute_graph(coords_cpu)

        if cache_key_norm is not None:
            self._graph_cache[cache_key_norm] = edge_index_cpu.cpu()

        return edge_index_cpu.to(device=coords.device)
