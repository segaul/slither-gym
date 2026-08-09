"""R1 regression (P0.6): the C4 density top-up must govern FLOOR food only.

Bug: the top-up compared food_density_per_1e6's target against alive_count(),
which includes corpse pellets. Under corpse_value_law='real' (P0.3, 20x victim
mass in pellets) a single big kill flooded the counter and suppressed floor
spawning until the corpse was eaten. The converse also held: the full-pool
eviction in FoodManager.spawn_at picked the global lowest-value pellet, so
corpse pellets could be evicted while floor food (replenishable by the top-up)
survived.

Fix: FoodManager tracks floor pellets separately (floor_count()); the density
top-up targets floor pellets only, and — density mode only — eviction removes
floor pellets before corpse pellets. Legacy (density off) behavior is
byte-identical, proven by golden fingerprints below captured on the pre-fix
code (commit 6edc2e1).
"""

import dataclasses
import hashlib
import math

import numpy as np
import pytest

from slither_gym.core.food import FoodManager
from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World


def _fingerprint(fm: FoodManager) -> str:
    """Full internal state of a FoodManager, hashed."""
    h = hashlib.sha256()
    h.update(fm._positions.tobytes())
    h.update(fm._values.tobytes())
    h.update(fm._alive.tobytes())
    h.update(fm._is_corpse.tobytes())
    h.update(np.int64(fm._count).tobytes())
    h.update(np.array(fm._free, dtype=np.int64).tobytes())
    return h.hexdigest()


def _density_config(**overrides: object) -> WorldConfig:
    base = dict(
        map_radius=1000.0,
        max_food=1024,
        food_density_per_1e6=62.0,
        food_refresh_interval=1,
    )
    base.update(overrides)
    return dataclasses.replace(WorldConfig(), **base)


# --------------------------------------------------------------------------
# Floor density is maintained while a huge corpse exists
# --------------------------------------------------------------------------


def test_r1_floor_topup_unaffected_by_large_corpse() -> None:
    c = _density_config()
    world = World(c, seed=0)
    target = world._target_food_count()
    assert target is not None and target > 0
    assert world._food.floor_count() == target

    # Drop a corpse worth far more than the whole floor budget (12.1/pellet
    # vs ~2 for floor food -> ~12x the ambient mass). Kept under the pool's
    # 25% free-slot reserve so raw capacity is not the binding constraint.
    n_corpse = 2 * target
    # Placed far from the origin so the collect_near below (radius 500 around
    # the origin) eats floor food only, not the corpse.
    for i in range(n_corpse):
        world._food.spawn_at(700.0 + float(i % 20), float(i // 20), 12.1, corpse=True)
    assert world._food.alive_count() == target + n_corpse

    # Eat some floor food, then let the top-up run.
    world._food.collect_near(0.0, 0.0, c.map_radius * 0.5)
    floor_after_eat = world._food.floor_count()
    assert floor_after_eat < target
    world.spawn_snake(0)
    world.step({0: (1.0, 0.0, False)})

    # Pre-fix, alive_count (floor + corpse) >> target meant ZERO floor top-up.
    # Post-fix the floor population is restored to the density target.
    assert world._food.floor_count() == target
    # And the corpse pellets are still extra mass on top of the floor budget.
    is_corpse = world._food.get_alive_is_corpse()
    assert int(np.count_nonzero(is_corpse)) > 0
    assert world._food.alive_count() > target


def test_r1_kill_via_world_path_does_not_starve_floor_food() -> None:
    """End-to-end: a real kill under corpse_value_law='real' must not
    suppress floor spawning."""
    c = _density_config(corpse_value_law="real", initial_mass=50.0)
    world = World(c, seed=3)
    target = world._target_food_count()
    assert target is not None

    world.spawn_snake(0)
    world.spawn_snake(1)
    corpse = world._snakes.kill(1, world._segments)
    assert len(corpse) > 0
    for fx, fy, fv in corpse:
        world._food.spawn_at(fx, fy, fv)

    # Consume a chunk of floor food, then step: top-up must restore it.
    world._food.collect_near(0.0, 0.0, c.map_radius * 0.4)
    world.step({0: (1.0, 0.0, False)})
    assert world._food.floor_count() == target


# --------------------------------------------------------------------------
# Eviction under pressure never prefers corpse pellets (density mode)
# --------------------------------------------------------------------------


def test_r1_corpse_pellets_not_evicted_under_pressure() -> None:
    c = _density_config(max_food=16)
    fm = FoodManager(c, np.random.default_rng(0))
    # Fill: 8 floor pellets (low value) + 8 corpse pellets (mixed values,
    # some LOWER than the floor pellets so the legacy global-argmin rule
    # would have evicted corpses first).
    for i in range(8):
        fm.spawn_at(float(i), 0.0, 5.0, corpse=False)
    for i in range(8):
        fm.spawn_at(float(i), 10.0, 1.0 + i, corpse=True)
    assert not fm._free

    # Overflow with more corpse pellets: every eviction must hit floor food.
    for i in range(8):
        fm.spawn_at(float(i), 20.0, 12.1, corpse=True)
    assert int(np.count_nonzero(fm.get_alive_is_corpse())) == 16
    assert fm.floor_count() == 0

    # Only corpses left: eviction falls back to lowest-value corpse
    # rather than failing.
    fm.spawn_at(0.0, 30.0, 12.1, corpse=True)
    assert fm.alive_count() == 16
    assert int(np.count_nonzero(fm.get_alive_is_corpse())) == 16


def test_r1_floor_count_bookkeeping_through_collect_and_evict() -> None:
    c = _density_config(max_food=32)
    fm = FoodManager(c, np.random.default_rng(1))
    fm.spawn_batch(20)
    assert fm.floor_count() == 20
    fm.spawn_at(0.0, 0.0, 12.1, corpse=True)
    assert fm.floor_count() == 20
    assert fm.alive_count() == 21
    collected = fm.collect_near(0.0, 0.0, 1e9)  # eat everything
    assert collected > 0
    assert fm.floor_count() == 0
    assert fm.alive_count() == 0


# --------------------------------------------------------------------------
# Legacy (density off) byte-identity
# --------------------------------------------------------------------------

# Golden fingerprints captured by running these exact scenarios on the
# pre-fix code (commit 6edc2e1). They cover the pool-overflow eviction path,
# collect_near, spawn_batch, and a world-level run with a corpse drop.
_GOLDEN_FM = "784c3e8e7056817bbc100e0e4e12621a8f1331abc4f1fd1ef603f5fc5650c22a"
_GOLDEN_WORLD = "90acd4a0400efb10ef92fb04302922c7ec34565942f0297d1b454725d1bbf204"


def test_r1_legacy_food_manager_byte_identical() -> None:
    cfg = dataclasses.replace(WorldConfig(), max_food=64)
    assert cfg.food_density_per_1e6 is None
    fm = FoodManager(cfg, np.random.default_rng(1234))
    fm.spawn_batch(60)  # reserve-limited
    vrng = np.random.default_rng(99)
    for _ in range(120):  # corpse drops overflowing the pool -> evictions
        fm.spawn_at(
            float(vrng.uniform(-100, 100)),
            float(vrng.uniform(-100, 100)),
            float(vrng.uniform(1.0, 30.0)),
            corpse=True,
        )
    fm.collect_near(0.0, 0.0, 50.0)
    fm.spawn_batch(20)
    assert _fingerprint(fm) == _GOLDEN_FM


def test_r1_legacy_world_run_byte_identical() -> None:
    c = dataclasses.replace(WorldConfig(), max_food=256)
    assert c.food_density_per_1e6 is None
    w = World(c, seed=7)
    w.spawn_snake(0)
    w.spawn_snake(1)
    for t in range(200):
        acts = {
            sid: (math.cos(t * 0.13 + sid), math.sin(t * 0.13 + sid), t % 17 == 0)
            for sid in (0, 1)
            if w._snakes.get_state(sid).alive
        }
        w.step(acts)
    alive = [sid for sid in (0, 1) if w._snakes.get_state(sid).alive]
    assert alive == [0, 1]  # scenario sanity: matches the golden capture
    corpse = w._snakes.kill(alive[0], w._segments)
    for fx, fy, fv in corpse:
        w._food.spawn_at(fx, fy, fv)
    for t in range(100):
        acts = {
            sid: (math.cos(t * 0.21 + sid), math.sin(t * 0.21 + sid), False)
            for sid in (0, 1)
            if w._snakes.get_state(sid).alive
        }
        w.step(acts)
    assert _fingerprint(w._food) == _GOLDEN_WORLD
