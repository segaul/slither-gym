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

    fm.spawn_at(50.0, 50.0, 2.0)
    collected = fm.collect_near(50.0, 50.0, 10.0)
    assert collected == 2.0

    # Food should be removed
    assert len(fm.get_alive_positions()) == 0


def test_collect_empty_area() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    fm = FoodManager(config, rng)

    fm.spawn_at(1000.0, 1000.0, 1.0)
    collected = fm.collect_near(0.0, 0.0, 10.0)
    assert collected == 0.0


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
