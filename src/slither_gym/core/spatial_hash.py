import numpy as np
from numpy.typing import NDArray


class SpatialHash:
    """
    Grid-based spatial index for fast "what segments are near this point?" queries.
    Rebuilt from scratch each tick.

    Perf (env-throughput batch, 2026-08-09): the rebuild was a Python loop
    appending one (seg_idx, owner) tuple per alive segment (~10-16k
    appends/tick at 64 sized bots). It is now a vectorized group-by: one
    stable argsort of packed cell keys, np.split into per-cell index/owner
    arrays. Bitwise-identical query results are guaranteed by construction:
      - alive_indices from np.where is ascending, and the STABLE sort keeps
        that ascending order within each cell — the same per-cell entry
        order the append loop produced;
      - query_near visits the 3x3 neighborhood in the same (dx, dy) order
        and concatenates per-cell arrays, so hits come back in the exact
        order the old per-entry loop yielded them (world.step breaks on the
        FIRST narrow-phase hit, so order is load-bearing);
      - the narrow phase casts candidate positions to float64 first, which
        reproduces the old `float(segments[i, 0]) - x` scalar arithmetic
        exactly (float32 -> float64 widening is exact, then all ops in
        float64, identical per element).
    Verified against the P0.6-style SHA-256 physics goldens
    (slither-rl scripts/golden_fingerprint.py).
    """

    def __init__(self, cell_size: float, bounds: float) -> None:
        self._cell_size = cell_size
        self._bounds = bounds
        # cell (cx, cy) -> (segment indices asc, owner snake ids), parallel arrays
        self._cells: dict[tuple[int, int], tuple[NDArray[np.int64], NDArray[np.int64]]] = {}
        self._segments: NDArray[np.float32] = np.zeros((0, 2), dtype=np.float32)

    def rebuild(
        self,
        segments: NDArray[np.float32],
        alive_mask: NDArray[np.bool_],
        snake_ids: NDArray[np.int32],
    ) -> None:
        """Clear grid and re-insert all alive segments."""
        self._cells.clear()
        self._segments = segments

        alive_indices = np.where(alive_mask)[0]
        if len(alive_indices) == 0:
            return

        cs = self._cell_size
        positions = segments[alive_indices]
        owners = snake_ids[alive_indices].astype(np.int64)
        alive_indices = alive_indices.astype(np.int64)

        # Same truncation-toward-zero convention as the query's int(x / cs).
        cx_arr = (positions[:, 0] / cs).astype(np.int32)
        cy_arr = (positions[:, 1] / cs).astype(np.int32)

        # Pack (cx, cy) into one int64 key; stable sort preserves the ascending
        # segment-index order within each cell.
        keys = (cx_arr.astype(np.int64) << 32) | (cy_arr.astype(np.int64) & 0xFFFFFFFF)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_idx = alive_indices[order]
        sorted_own = owners[order]

        boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        cells = self._cells
        starts_list: list[int] = starts.tolist()
        ends_list: list[int] = starts_list[1:] + [len(sorted_keys)]
        kx: list[int] = cx_arr[order][starts].tolist()
        ky: list[int] = cy_arr[order][starts].tolist()
        for a, b, gx, gy in zip(starts_list, ends_list, kx, ky):
            cells[(gx, gy)] = (sorted_idx[a:b], sorted_own[a:b])

    def query_near(
        self,
        x: float,
        y: float,
        radius: float,
        exclude_snake_id: int,
    ) -> list[tuple[int, int]]:
        """
        Returns list of (segment_global_index, snake_id) within radius of (x, y),
        excluding segments belonging to exclude_snake_id.
        """
        assert radius <= self._cell_size, (
            f"Query radius {radius} exceeds cell_size {self._cell_size}"
        )

        cs = self._cell_size
        cx = int(x / cs)
        cy = int(y / cs)
        r_sq = radius * radius
        cells = self._cells

        idx_parts: list[NDArray[np.int64]] = []
        own_parts: list[NDArray[np.int64]] = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = cells.get((cx + dx, cy + dy))
                if cell is None:
                    continue
                idx_parts.append(cell[0])
                own_parts.append(cell[1])
        if not idx_parts:
            return []

        idx = np.concatenate(idx_parts) if len(idx_parts) > 1 else idx_parts[0]
        own = np.concatenate(own_parts) if len(own_parts) > 1 else own_parts[0]
        keep = own != exclude_snake_id
        idx = idx[keep]
        if idx.size == 0:
            return []
        own = own[keep]

        # float64 cast reproduces the old float(np.float32) scalar math exactly.
        p = self._segments[idx].astype(np.float64)
        sx = p[:, 0] - x
        sy = p[:, 1] - y
        hit = (sx * sx + sy * sy) <= r_sq
        if not np.any(hit):
            return []
        return list(zip(idx[hit].tolist(), own[hit].tolist()))
