import numpy as np
from numpy.typing import NDArray


def build_circular_mask(size: int) -> NDArray[np.bool_]:
    """Precompute a circular mask for the minimap grid."""
    half = size / 2.0
    y, x = np.mgrid[:size, :size]
    dist = np.sqrt((x - half + 0.5) ** 2 + (y - half + 0.5) ** 2)
    mask: NDArray[np.bool_] = dist <= half
    return mask


def compute_minimap(
    snake_positions: NDArray[np.float32],
    snake_masses: NDArray[np.float32],
    map_radius: float,
    grid_size: int = 32,
) -> NDArray[np.float32]:
    """
    Build a circular minimap showing snake mass density.
    Returns (grid_size, grid_size) float32 array.
    Cells outside the map circle are zero.
    Each cell contains log(1 + total_mass) of snakes in that region.
    """
    minimap = np.zeros((grid_size, grid_size), dtype=np.float32)

    if len(snake_positions) == 0:
        return minimap

    r = map_radius
    gx = ((snake_positions[:, 0] + r) / (2 * r) * grid_size).astype(np.int32)
    gy = ((snake_positions[:, 1] + r) / (2 * r) * grid_size).astype(np.int32)

    valid = (gx >= 0) & (gx < grid_size) & (gy >= 0) & (gy < grid_size)
    gx = gx[valid]
    gy = gy[valid]
    m = snake_masses[valid]

    np.add.at(minimap, (gy, gx), m)

    minimap = np.log1p(minimap)

    mask = build_circular_mask(grid_size)
    minimap[~mask] = 0.0

    return minimap
