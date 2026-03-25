import math
import time

import numpy as np

from slither_gym.core.snake import SnakeManager, compute_segment_radius, compute_turn_rate
from slither_gym.core.types import WorldConfig


def _make_segments(config: WorldConfig) -> np.ndarray:
    total = config.max_snakes * config.max_segments_per_snake
    return np.zeros((total, 2), dtype=np.float32)


def test_spawn_horizontal_line() -> None:
    config = WorldConfig()
    mgr = SnakeManager(config)
    segments = _make_segments(config)

    state = mgr.spawn(0, 0.0, 0.0, 0.0, segments)  # facing right

    assert state.alive
    assert state.mass == config.initial_mass
    assert state.segment_count == config.initial_segments

    # Head at (0, 0)
    assert segments[0, 0] == 0.0
    assert segments[0, 1] == 0.0
    # Segments trail to the left
    for i in range(1, config.initial_segments):
        assert segments[i, 0] < segments[i - 1, 0]
        np.testing.assert_allclose(segments[i, 1], 0.0, atol=1e-5)


def test_move_straight_ahead() -> None:
    config = WorldConfig()
    mgr = SnakeManager(config)
    segments = _make_segments(config)

    mgr.spawn(0, 0.0, 0.0, 0.0, segments)
    old_head_x = segments[0, 0]

    mgr.move(0, 1.0, 0.0, False, segments)  # target = right (cos=1, sin=0)

    assert segments[0, 0] > old_head_x
    np.testing.assert_allclose(segments[0, 0], config.base_speed, atol=1e-5)


def test_turn_rate_clamping() -> None:
    config = WorldConfig()
    mgr = SnakeManager(config)
    segments = _make_segments(config)

    state = mgr.spawn(0, 0.0, 0.0, 0.0, segments)  # facing right (angle=0)
    turn_rate = state.turn_rate

    # Target 90 degrees away (up)
    mgr.move(0, 0.0, 1.0, False, segments)

    new_state = mgr.get_state(0)
    # Should have turned by exactly turn_rate, not jumped to pi/2
    np.testing.assert_allclose(new_state.angle, turn_rate, atol=1e-5)


def test_boost() -> None:
    config = WorldConfig()
    mgr = SnakeManager(config)
    segments = _make_segments(config)

    state = mgr.spawn(0, 0.0, 0.0, 0.0, segments)
    # Give snake extra mass so it can boost (floor is initial_mass)
    mgr.grow(0, 20.0, segments)
    state = mgr.get_state(0)
    mass_before = state.mass

    mgr.move(0, 1.0, 0.0, True, segments)

    state = mgr.get_state(0)
    # Head should advance by boost_speed
    assert state.speed == config.boost_speed
    # Mass should decrease
    assert state.mass < mass_before
    np.testing.assert_allclose(
        state.mass, mass_before - config.boost_mass_cost_per_tick, atol=1e-5
    )


def test_boost_floor() -> None:
    """Can't boost below initial mass."""
    config = WorldConfig()
    mgr = SnakeManager(config)
    segments = _make_segments(config)

    state = mgr.spawn(0, 0.0, 0.0, 0.0, segments)
    # At initial mass, boost should not activate
    mgr.move(0, 1.0, 0.0, True, segments)
    state = mgr.get_state(0)
    assert not state.boosting
    assert state.speed == config.base_speed


def test_grow() -> None:
    config = WorldConfig()
    mgr = SnakeManager(config)
    segments = _make_segments(config)

    state = mgr.spawn(0, 0.0, 0.0, 0.0, segments)
    old_count = state.segment_count

    mgr.grow(0, 5.0, segments)

    state = mgr.get_state(0)
    assert state.mass == config.initial_mass + 5.0

    # Segments grow during move(), not grow()
    for _ in range(20):
        mgr.move(0, 1.0, 0.0, False, segments)
    state = mgr.get_state(0)
    assert state.segment_count > old_count


def test_kill_returns_corpse() -> None:
    config = WorldConfig()
    mgr = SnakeManager(config)
    segments = _make_segments(config)

    mgr.spawn(0, 100.0, 200.0, 0.0, segments)

    corpse = mgr.kill(0, segments)

    state = mgr.get_state(0)
    assert not state.alive
    assert len(corpse) == config.initial_segments
    # Corpse positions should match segment positions
    for i, (cx, cy, cv) in enumerate(corpse):
        np.testing.assert_allclose(cx, segments[i, 0], atol=1e-5)
        np.testing.assert_allclose(cy, segments[i, 1], atol=1e-5)
        assert cv >= config.corpse_food_base


def test_move_benchmark() -> None:
    config = WorldConfig(initial_segments=100, initial_mass=100.0)
    mgr = SnakeManager(config)
    segments = _make_segments(config)
    mgr.spawn(0, 0.0, 0.0, 0.0, segments)

    # Warmup
    for _ in range(100):
        mgr.move(0, 1.0, 0.0, False, segments)

    start = time.perf_counter()
    n = 1000
    for _ in range(n):
        mgr.move(0, 1.0, 0.0, False, segments)
    elapsed = (time.perf_counter() - start) / n * 1000  # ms

    assert elapsed < 0.15, f"move() too slow: {elapsed:.4f}ms (need <0.15ms)"


def test_scaling_functions() -> None:
    config = WorldConfig()

    # At initial mass: near min radius, near max turn rate
    r_init = compute_segment_radius(config.initial_mass, config)
    t_init = compute_turn_rate(config.initial_mass, config)
    assert abs(r_init - config.min_segment_radius) < 1.0
    assert abs(t_init - config.max_turn_rate) < 0.01

    # At max mass: near max radius, near min turn rate
    r_max = compute_segment_radius(config.max_mass, config)
    t_max = compute_turn_rate(config.max_mass, config)
    np.testing.assert_allclose(r_max, config.max_segment_radius, atol=0.01)
    np.testing.assert_allclose(t_max, config.min_turn_rate, atol=0.001)

    # Beyond max mass: clamped
    r_over = compute_segment_radius(config.max_mass * 2, config)
    np.testing.assert_allclose(r_over, config.max_segment_radius, atol=0.01)

    # Monotonic
    prev_r = 0.0
    prev_t = 1.0
    for mass in [10, 100, 1000, 10000, 40000]:
        r = compute_segment_radius(float(mass), config)
        t = compute_turn_rate(float(mass), config)
        assert r >= prev_r
        assert t <= prev_t
        prev_r = r
        prev_t = t
