import math

import numpy as np
from numpy.typing import NDArray

from slither_gym.core.types import WorldConfig

# Pellet-grid cell size (u). Only a perf knob: queries are bbox-based (any
# radius), and every arithmetic result is independent of it. 256 keeps a
# collect-radius query (~<100 u) at <=9 cells and a 500 u obs query at ~9-16.
_FOOD_CELL = 256.0


class FoodManager:
    """
    Manages food pellets in a pre-allocated array.
    Uses a free-list pattern for O(1) spawn/despawn without reallocation.

    Perf (env-throughput batch, 2026-08-09): collect_near used to scan the
    ENTIRE pool (81920 slots at the eval-matched config) per snake per tick.
    Pellets are now also indexed in a cell grid; collect_near and
    query_candidates gather only nearby cells and then run the ORIGINAL
    full-scan arithmetic on that candidate subset, sorted ascending by global
    index — the same subset in the same order the full-pool boolean masks
    produced, so distances (float32, same elementwise ops), the pairwise
    np.sum of collected values, and the free-list push order are all
    bitwise-identical. Cells are derived from the STORED float32 positions
    (not the pre-store float64 spawn coords) and the query bbox is padded by
    one cell, so float32 rounding at a cell boundary can never hide a pellet
    the full scan would have hit. Verified vs the SHA-256 physics goldens.
    """

    def __init__(self, config: WorldConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

        self._positions = np.zeros((config.max_food, 2), dtype=np.float32)
        self._values = np.zeros(config.max_food, dtype=np.float32)
        self._alive = np.zeros(config.max_food, dtype=np.bool_)
        self._is_corpse = np.zeros(config.max_food, dtype=np.bool_)
        self._count: int = 0
        # Floor (non-corpse) pellets only. R1: the C4 density top-up must
        # target ambient floor food, not floor + corpse — otherwise a big
        # corpse drop (20x victim mass under corpse_value_law='real')
        # suppresses floor spawning until the corpse is eaten.
        self._floor_count: int = 0
        self._free: list[int] = list(range(config.max_food))
        # Cell grid over ALIVE pellets: (cx, cy) -> set of pellet indices.
        # Sets are fine (unordered): every consumer sorts candidates ascending.
        self._grid: dict[tuple[int, int], set[int]] = {}
        # Registered cell per alive slot, so removal never re-derives the key.
        self._cell_of: dict[int, tuple[int, int]] = {}

    def _grid_add(self, idx: int) -> None:
        cs = _FOOD_CELL
        key = (
            math.floor(float(self._positions[idx, 0]) / cs),
            math.floor(float(self._positions[idx, 1]) / cs),
        )
        self._grid.setdefault(key, set()).add(idx)
        self._cell_of[idx] = key

    def _grid_remove(self, idx: int) -> None:
        key = self._cell_of.pop(idx)
        cell = self._grid[key]
        cell.discard(idx)
        if not cell:
            del self._grid[key]

    def _candidates(self, x: float, y: float, radius: float) -> NDArray[np.int64]:
        """Alive pellet indices (ascending) from all cells overlapping the
        radius-bbox around (x, y), padded one cell against float32 boundary
        rounding. Superset of every pellet the full scan could hit."""
        cs = _FOOD_CELL
        cx0 = math.floor((x - radius) / cs) - 1
        cx1 = math.floor((x + radius) / cs) + 1
        cy0 = math.floor((y - radius) / cs) - 1
        cy1 = math.floor((y + radius) / cs) + 1
        grid = self._grid
        found: list[int] = []
        # Huge radii (tests use 1e9) would make the bbox loop iterate millions
        # of empty cells; when the bbox has more cells than exist, walk the
        # occupied cells instead. Same candidate set either way.
        if (cx1 - cx0 + 1) * (cy1 - cy0 + 1) >= len(grid):
            for (cx, cy), cell in grid.items():
                if cx0 <= cx <= cx1 and cy0 <= cy <= cy1:
                    found.extend(cell)
        else:
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    cell = grid.get((cx, cy))
                    if cell:
                        found.extend(cell)
        if not found:
            return np.zeros(0, dtype=np.int64)
        cand = np.array(found, dtype=np.int64)
        cand.sort()
        return cand

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
            self.spawn_at(x, y, self._sample_value(), corpse=False)

    def _sample_value(self) -> float:
        """One floor-pellet value, per config.food_value_law.

        "legacy" draws exactly one uniform(min, max) -- the identical RNG call
        the pre-P0.3 code made inline, so existing seeded runs are
        byte-identical. "real" draws the measured cluster+tail mixture
        (min 3.0, p25 4.8, p50 5.2, p75 6.2, tail to 14.2, mean 6.25 --
        fit documented in core/realism.py, FOOD_BULK_* note).
        """
        c = self._config
        if c.food_value_law == "legacy":
            # Most food is small, occasionally larger
            return float(self._rng.uniform(c.food_value_min, c.food_value_max))
        if c.food_value_law == "real":
            if self._rng.uniform() < c.food_tail_weight:
                # High-value tail (the 10-14.2 corpse-like cluster).
                return float(self._rng.uniform(c.food_tail_lo, c.food_tail_hi))
            # Bulk cluster at ~5.2. Clipped to the measured support floor (3.0)
            # and to the tail's lower edge so bulk and tail stay disjoint;
            # both clips are ~4-sigma events with negligible mass effect.
            v = float(self._rng.lognormal(c.food_bulk_log_mu, c.food_bulk_log_sigma))
            return min(max(v, 3.0), c.food_tail_lo)
        raise ValueError(f"unknown food_value_law: {c.food_value_law!r}")

    def spawn_at(self, x: float, y: float, value: float, corpse: bool = True) -> None:
        """Spawn a single pellet at a specific position.
        corpse=True for corpse drops (default), False for regular food.
        If pool is full, evicts the lowest-value alive food to make room."""
        if not self._free:
            # Evict to make room. R1: when the density feature is on
            # (food_density_per_1e6 set), corpse pellets are protected — they
            # are extra mass in flight, not ambient budget, and the density
            # top-up will restore any evicted floor pellet next refresh. So
            # evict the lowest-value FLOOR pellet first, falling back to the
            # lowest-value corpse pellet only if no floor food is alive.
            # When the density feature is off we keep the legacy rule
            # (global lowest-value eviction, corpse or floor) so seeded
            # legacy runs stay byte-identical (see test_r1_density.py).
            if self._config.food_density_per_1e6 is not None:
                candidates = np.where(self._alive & ~self._is_corpse)[0]
                if len(candidates) == 0:
                    candidates = np.where(self._alive)[0]
            else:
                candidates = np.where(self._alive)[0]
            if len(candidates) == 0:
                return
            min_idx = int(candidates[np.argmin(self._values[candidates])])
            self._alive[min_idx] = False
            self._count -= 1
            if not self._is_corpse[min_idx]:
                self._floor_count -= 1
            self._free.append(min_idx)
            self._grid_remove(min_idx)
        idx = self._free.pop()
        self._positions[idx, 0] = x
        self._positions[idx, 1] = y
        self._values[idx] = value
        self._alive[idx] = True
        self._is_corpse[idx] = corpse
        self._count += 1
        if not corpse:
            self._floor_count += 1
        self._grid_add(idx)

    def collect_near(self, x: float, y: float, radius: float) -> float:
        """Remove all food within radius of (x, y). Returns total value collected.

        Bitwise contract: identical to the historical full-pool scan — the
        candidate subset is ascending-ordered and a superset of any possible
        hit, the distance test is the same float32 expression, and the value
        sum runs over the same hit sequence (same pairwise summation)."""
        if self._count == 0:
            return 0.0

        cand = self._candidates(x, y, radius)
        if cand.size == 0:
            return 0.0

        pos = self._positions[cand]
        dx = pos[:, 0] - x
        dy = pos[:, 1] - y
        dist_sq = dx * dx + dy * dy
        hit = self._alive[cand] & (dist_sq < radius * radius)

        if not np.any(hit):
            return 0.0

        hit_indices = cand[hit]
        total = float(np.sum(self._values[hit_indices]))
        self._alive[hit_indices] = False
        self._count -= len(hit_indices)
        self._floor_count -= int(np.count_nonzero(~self._is_corpse[hit_indices]))
        self._free.extend(hit_indices.tolist())
        for i in hit_indices.tolist():
            self._grid_remove(i)

        return total

    def query_candidates(self, x: float, y: float, radius: float) -> NDArray[np.int64]:
        """READ-ONLY: ascending global indices of alive pellets in the padded
        radius-bbox around (x, y) — a superset of everything within `radius`.
        For observation builders that then reproduce the exact full-scan
        arithmetic on the subset (see rl/env_gym bot-obs fast path)."""
        if self._count == 0:
            return np.zeros(0, dtype=np.int64)
        return self._candidates(x, y, radius)

    def alive_count(self) -> int:
        """Number of pellets currently alive (floor food + corpse drops)."""
        return self._count

    def floor_count(self) -> int:
        """Number of FLOOR (non-corpse) pellets currently alive.

        R1: this is what the C4 density top-up must compare against
        food_density_per_1e6's target — corpse pellets are transient extra
        mass, not part of the ambient floor-food budget."""
        return self._floor_count

    def get_alive_positions(self) -> NDArray[np.float32]:
        result: NDArray[np.float32] = self._positions[self._alive]
        return result

    def get_alive_values(self) -> NDArray[np.float32]:
        result: NDArray[np.float32] = self._values[self._alive]
        return result

    def get_alive_is_corpse(self) -> NDArray[np.bool_]:
        result: NDArray[np.bool_] = self._is_corpse[self._alive]
        return result
