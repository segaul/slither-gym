"""INFRA-batched-hotspots gate (food half): collect_near_batch is BITWISE identical to
the sequential per-snake collect_near loop — per-snake (total, corpse) floats equal by
==, and the pool's full final state (alive mask, counts, free-list ORDER, grid, floor
count) identical — over randomized dense/sparse geometry including radius-edge pellets,
overlapping snakes contending for the same pellets, corpse mixes, and the huge-radius
candidate fallback path."""
import copy

import numpy as np
import pytest

from slither_gym.core import food as food_mod
from slither_gym.core.food import FoodManager
from slither_gym.core.types import WorldConfig


def _mk(seed: int, max_food: int = 4096) -> FoodManager:
    cfg = WorldConfig(max_food=max_food, map_radius=15000.0)
    fm = FoodManager(cfg, np.random.default_rng(seed))
    return fm


def _fill(fm: FoodManager, rng: np.random.Generator, n: int, spread: float) -> None:
    for _ in range(n):
        x = float(rng.uniform(-spread, spread))
        y = float(rng.uniform(-spread, spread))
        fm.spawn_at(x, y, float(rng.uniform(0.5, 14.2)), corpse=bool(rng.uniform() < 0.3))


def _state(fm: FoodManager) -> tuple:
    return (
        fm._alive.tobytes(),
        fm._count,
        fm._floor_count,
        list(fm._free),
        {k: sorted(v) for k, v in fm._grid.items()},
        fm._values.tobytes(),
    )


@pytest.mark.skipif(food_mod.FOOD_KERNEL == "python", reason="numba not installed")
def test_batch_bitwise_identical_randomized() -> None:
    rng = np.random.default_rng(0)
    for case in range(300):
        seed = int(rng.integers(1 << 30))
        spread = float(rng.choice([400.0, 3000.0, 14000.0]))
        n_pellets = int(rng.integers(1, 1200))
        n_snakes = int(rng.integers(1, 40))

        a, b = _mk(seed), _mk(seed)
        fill_rng = np.random.default_rng(seed + 1)
        _fill(a, fill_rng, n_pellets, spread)
        _fill(b, np.random.default_rng(seed + 1), n_pellets, spread)

        q = np.random.default_rng(seed + 2)
        xs, ys, rads = [], [], []
        for s in range(n_snakes):
            if s > 0 and q.uniform() < 0.3:
                # contention: sit on a previous snake's spot
                j = int(q.integers(0, s))
                xs.append(xs[j] + float(q.uniform(-5, 5)))
                ys.append(ys[j] + float(q.uniform(-5, 5)))
            else:
                xs.append(float(q.uniform(-spread, spread)))
                ys.append(float(q.uniform(-spread, spread)))
            rads.append(float(q.choice([8.0, 35.0, 120.0, 1e9])))  # 1e9 -> fallback path

        ref = [a.collect_near(x, y, r) for x, y, r in zip(xs, ys, rads)]
        got = b.collect_near_batch(xs, ys, rads)

        assert ref == got, f"case {case}: results diverged"
        assert _state(a) == _state(b), f"case {case}: pool state diverged"


@pytest.mark.skipif(food_mod.FOOD_KERNEL == "python", reason="numba not installed")
def test_batch_radius_edge_pellets() -> None:
    # Pellets planted AT the collect radius probe float32 boundary rounding: the
    # scalar->float32 casts in the kernel must round exactly as numpy's NEP-50 ops.
    rng = np.random.default_rng(7)
    for case in range(2000):
        a, b = _mk(case), _mk(case)
        x = float(rng.uniform(-10000, 10000))
        y = float(rng.uniform(-10000, 10000))
        r = float(rng.uniform(5.0, 150.0))
        for _ in range(8):
            ang = float(rng.uniform(0, 2 * np.pi))
            eps = float(rng.uniform(-1e-3, 1e-3))
            px = x + (r + eps) * np.cos(ang)
            py = y + (r + eps) * np.sin(ang)
            a.spawn_at(px, py, 5.0, corpse=False)
            b.spawn_at(px, py, 5.0, corpse=False)
        ref = [a.collect_near(x, y, r)]
        got = b.collect_near_batch([x], [y], [r])
        assert ref == got and _state(a) == _state(b), f"case {case}"


@pytest.mark.skipif(food_mod.FOOD_KERNEL == "python", reason="numba not installed")
def test_batch_empty_pool_and_no_hits() -> None:
    fm = _mk(1)
    assert fm.collect_near_batch([0.0], [0.0], [50.0]) == [(0.0, 0.0)]
    fm.spawn_at(5000.0, 5000.0, 5.0, corpse=False)
    assert fm.collect_near_batch([0.0], [0.0], [50.0]) == [(0.0, 0.0)]
    assert fm.alive_count() == 1
