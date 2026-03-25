import numpy as np
from numpy.typing import NDArray

from slither_gym.core.types import WorldConfig


class FoodManager:
    """
    Manages food pellets in a pre-allocated array.
    Uses a free-list pattern for O(1) spawn/despawn without reallocation.
    """

    def __init__(self, config: WorldConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

        self._positions = np.zeros((config.max_food, 2), dtype=np.float32)
        self._values = np.zeros(config.max_food, dtype=np.float32)
        self._alive = np.zeros(config.max_food, dtype=np.bool_)
        self._count: int = 0
        self._free: list[int] = list(range(config.max_food))

    def spawn_batch(self, count: int) -> None:
        """Spawn count pellets at random positions within map bounds.
        Stops early if pool is >75% full to reserve space for corpse food."""
        r = self._config.map_radius
        reserve = self._config.max_food // 4
        for _ in range(count):
            if len(self._free) < reserve:
                return
            angle = self._rng.uniform(0, 2 * np.pi)
            dist = r * np.sqrt(self._rng.uniform(0, 1))
            x = float(dist * np.cos(angle))
            y = float(dist * np.sin(angle))
            # Most food is small, occasionally larger
            value = float(self._rng.uniform(self._config.food_value_min, self._config.food_value_max))
            self.spawn_at(x, y, value)

    def spawn_at(self, x: float, y: float, value: float) -> None:
        """Spawn a single pellet at a specific position (for corpse drops).
        If pool is full, evicts the lowest-value alive food to make room."""
        if not self._free:
            # Evict: find the lowest-value alive food and remove it
            alive_indices = np.where(self._alive)[0]
            if len(alive_indices) == 0:
                return
            min_idx = int(alive_indices[np.argmin(self._values[alive_indices])])
            self._alive[min_idx] = False
            self._count -= 1
            self._free.append(min_idx)
        idx = self._free.pop()
        self._positions[idx, 0] = x
        self._positions[idx, 1] = y
        self._values[idx] = value
        self._alive[idx] = True
        self._count += 1

    def collect_near(self, x: float, y: float, radius: float) -> float:
        """Remove all food within radius of (x, y). Returns total value collected."""
        if self._count == 0:
            return 0.0

        dx = self._positions[:, 0] - x
        dy = self._positions[:, 1] - y
        dist_sq = dx * dx + dy * dy
        hit = self._alive & (dist_sq < radius * radius)

        if not np.any(hit):
            return 0.0

        total = float(np.sum(self._values[hit]))
        hit_indices = np.where(hit)[0]
        self._alive[hit_indices] = False
        self._count -= len(hit_indices)
        self._free.extend(hit_indices.tolist())

        return total

    def get_alive_positions(self) -> NDArray[np.float32]:
        result: NDArray[np.float32] = self._positions[self._alive]
        return result

    def get_alive_values(self) -> NDArray[np.float32]:
        result: NDArray[np.float32] = self._values[self._alive]
        return result
