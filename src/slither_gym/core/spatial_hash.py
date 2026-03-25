import numpy as np
from numpy.typing import NDArray


class SpatialHash:
    """
    Grid-based spatial index for fast "what segments are near this point?" queries.
    Rebuilt from scratch each tick.
    """

    def __init__(self, cell_size: float, bounds: float) -> None:
        self._cell_size = cell_size
        self._bounds = bounds
        self._cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
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
        owners = snake_ids[alive_indices]

        cx_arr = (positions[:, 0] / cs).astype(np.int32)
        cy_arr = (positions[:, 1] / cs).astype(np.int32)

        cells = self._cells
        for i in range(len(alive_indices)):
            key = (int(cx_arr[i]), int(cy_arr[i]))
            entry = (int(alive_indices[i]), int(owners[i]))
            if key in cells:
                cells[key].append(entry)
            else:
                cells[key] = [entry]

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
        segments = self._segments

        result: list[tuple[int, int]] = []
        cells = self._cells

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = cells.get((cx + dx, cy + dy))
                if cell is None:
                    continue
                for seg_idx, sid in cell:
                    if sid == exclude_snake_id:
                        continue
                    sx = float(segments[seg_idx, 0]) - x
                    sy = float(segments[seg_idx, 1]) - y
                    if sx * sx + sy * sy <= r_sq:
                        result.append((seg_idx, sid))

        return result
