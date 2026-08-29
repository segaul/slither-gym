import time

import numpy as np

from slither_gym.core.food import FoodManager
from slither_gym.core.types import WorldConfig


def test_spawn_within_bounds() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    fm = FoodManager(config, rng)

    fm.spawn_batch(100)

    positions = fm.get_alive_positions()
    assert len(positions) == 100
    dists = np.sqrt(np.sum(positions * positions, axis=1))
    assert np.all(dists <= config.map_radius)


def test_spawn_at_and_collect() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    fm = FoodManager(config, rng)

    # spawn_at defaults to corpse=True, so both halves of the split should see it.
    fm.spawn_at(50.0, 50.0, 2.0)
    collected, corpse = fm.collect_near(50.0, 50.0, 10.0)
    assert collected == 2.0
    assert corpse == 2.0

    # Food should be removed
    assert len(fm.get_alive_positions()) == 0


def test_collect_empty_area() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    fm = FoodManager(config, rng)

    fm.spawn_at(1000.0, 1000.0, 1.0)
    collected, corpse = fm.collect_near(0.0, 0.0, 10.0)
    assert collected == 0.0
    assert corpse == 0.0


def test_spawn_at_capacity() -> None:
    config = WorldConfig(max_food=10)
    rng = np.random.default_rng(42)
    fm = FoodManager(config, rng)

    for i in range(10):
        fm.spawn_at(float(i), 0.0, 1.0)

    # Should not crash at capacity
    fm.spawn_batch(5)
    assert len(fm.get_alive_positions()) == 10


def test_collect_near_benchmark() -> None:
    config = WorldConfig(max_food=1024)
    rng = np.random.default_rng(42)
    fm = FoodManager(config, rng)

    fm.spawn_batch(1024)

    # Warmup
    for _ in range(100):
        fm.collect_near(0.0, 0.0, 10.0)

    # Re-spawn for benchmark
    fm2 = FoodManager(WorldConfig(max_food=1024), rng)
    fm2.spawn_batch(1024)

    start = time.perf_counter()
    n = 1000
    for _ in range(n):
        fm2.collect_near(500.0, 500.0, 10.0)
    elapsed = (time.perf_counter() - start) / n * 1000

    assert elapsed < 0.02, f"collect_near too slow: {elapsed:.4f}ms (need <0.02ms)"


def test_collect_near_splits_corpse_from_floor() -> None:
    """collect_near must report floor and corpse intake separately.

    The `_is_corpse` flag was tracked but never read, so `remains_eaten` — documented
    as "mass gained from corpse food specifically" — actually counted every pellet.
    That double-paid floor food and made the corpse premium (D1) inexpressible.
    """
    config = WorldConfig()
    fm = FoodManager(config, np.random.default_rng(0))

    fm.spawn_at(0.0, 0.0, 3.0, corpse=False)
    fm.spawn_at(1.0, 1.0, 5.0, corpse=True)
    total, corpse = fm.collect_near(0.0, 0.0, 10.0)
    assert total == 8.0
    assert corpse == 5.0


def test_alive_floor_count_excludes_corpse() -> None:
    """R1: the density top-up must not let corpse drops starve floor spawning."""
    config = WorldConfig()
    fm = FoodManager(config, np.random.default_rng(0))

    fm.spawn_at(0.0, 0.0, 1.0, corpse=False)
    for i in range(5):
        fm.spawn_at(float(i), 0.0, 2.0, corpse=True)
    assert fm.alive_count() == 6
    assert fm.alive_floor_count() == 1


def test_spawn_mass_law_default_is_fixed() -> None:
    """D6 must be opt-in: legacy configs spawn every snake at initial_mass."""
    from slither_gym.core.world import World

    cfg = WorldConfig(max_snakes=4)
    assert cfg.spawn_mass_law == "fixed"
    w = World(cfg, 0)
    for i in range(4):
        w.spawn_snake(i, sample_mass=True)
    masses = {s.mass for s in w.get_snake_states().values()}
    assert masses == {cfg.initial_mass}


def test_spawn_mass_law_real_sct_is_heavy_tailed() -> None:
    """Fit targets the measured real distribution: sct p10 2 / p50 22 / p90 178."""
    from slither_gym.core.world import World

    cfg = WorldConfig(max_snakes=4, spawn_mass_law="real_sct")
    w = World(cfg, 0)
    draws = sorted(w.sample_spawn_mass() for _ in range(20000))
    p50 = draws[len(draws) // 2]
    p90 = draws[int(0.9 * len(draws))]
    assert 18.0 < p50 < 27.0, p50
    assert 140.0 < p90 < 220.0, p90
    assert draws[0] >= cfg.spawn_mass_min
    assert draws[-1] <= cfg.spawn_mass_max
