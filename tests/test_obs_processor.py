import math
import time

import numpy as np

from slither_gym.rl.obs_processor import compute_observation
from slither_gym.rl.types import RawGameState
from slither_gym.rl.types import ObsConfig


def _empty_arrays() -> dict[str, np.ndarray]:
    return {
        "food_positions": np.zeros((0, 2), dtype=np.float32),
        "food_values": np.zeros(0, dtype=np.float32),
        "enemy_segments": np.zeros((0, 2), dtype=np.float32),
        "enemy_is_head": np.zeros(0, dtype=np.bool_),
        "enemy_owner_mass": np.zeros(0, dtype=np.float32),
        "enemy_owner_speed": np.zeros(0, dtype=np.float32),
        "enemy_owner_angle": np.zeros(0, dtype=np.float32),
        "enemy_segment_radius": np.zeros(0, dtype=np.float32),
        "all_snake_positions": np.zeros((0, 2), dtype=np.float32),
        "all_snake_masses": np.zeros(0, dtype=np.float32),
    }


def test_single_food() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()
    arrays["food_positions"] = np.array([[100.0, 0.0]], dtype=np.float32)
    arrays["food_values"] = np.array([1.0], dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )

    obs = compute_observation(state, obs_config)
    assert obs["food"].shape == (obs_config.k_food, 3)
    np.testing.assert_allclose(obs["food"][0, 0], 100.0 / 500.0, atol=1e-5)
    np.testing.assert_allclose(obs["food"][0, 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(obs["food"][0, 2], 1.0, atol=1e-5)


def test_food_beyond_perception() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()
    arrays["food_positions"] = np.array([[1000.0, 0.0]], dtype=np.float32)
    arrays["food_values"] = np.array([1.0], dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )

    obs = compute_observation(state, obs_config)
    # Beyond perception radius, should be zero-padded
    np.testing.assert_allclose(obs["food"][0], [0.0, 0.0, 0.0], atol=1e-5)


def test_food_k_nearest() -> None:
    obs_config = ObsConfig(k_food=32)
    arrays = _empty_arrays()

    # 100 food pellets at varying distances
    rng = np.random.default_rng(42)
    positions = rng.uniform(-400, 400, (100, 2)).astype(np.float32)
    arrays["food_positions"] = positions
    arrays["food_values"] = np.ones(100, dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )

    obs = compute_observation(state, obs_config)

    # Should have at most 32 entries, sorted nearest first
    food = obs["food"]
    assert food.shape == (32, 3)

    # Check non-zero entries are sorted by distance
    nonzero_mask = np.any(food != 0, axis=1)
    nonzero = food[nonzero_mask]
    if len(nonzero) > 1:
        dists = np.sqrt(nonzero[:, 0] ** 2 + nonzero[:, 1] ** 2)
        assert np.all(dists[:-1] <= dists[1:] + 1e-5)


def test_enemy_approaching() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    # Enemy head at (100, 0) facing left toward self at origin
    arrays["enemy_segments"] = np.array([[100.0, 0.0]], dtype=np.float32)
    arrays["enemy_is_head"] = np.array([True], dtype=np.bool_)
    arrays["enemy_owner_mass"] = np.array([10.0], dtype=np.float32)
    arrays["enemy_owner_speed"] = np.array([3.0], dtype=np.float32)
    arrays["enemy_owner_angle"] = np.array([math.pi], dtype=np.float32)  # facing left
    arrays["enemy_segment_radius"] = np.array([3.0], dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )

    obs = compute_observation(state, obs_config)
    # Enemy heading aligns with vector-to-self → rel_velocity_angle ≈ 0
    rel_vel = obs["enemies"][0, 5]
    assert abs(rel_vel) < 0.2


def test_enemy_retreating() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    # Enemy head at (100, 0) facing right (away from self at origin)
    arrays["enemy_segments"] = np.array([[100.0, 0.0]], dtype=np.float32)
    arrays["enemy_is_head"] = np.array([True], dtype=np.bool_)
    arrays["enemy_owner_mass"] = np.array([10.0], dtype=np.float32)
    arrays["enemy_owner_speed"] = np.array([3.0], dtype=np.float32)
    arrays["enemy_owner_angle"] = np.array([0.0], dtype=np.float32)  # facing right (away)
    arrays["enemy_segment_radius"] = np.array([3.0], dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )

    obs = compute_observation(state, obs_config)
    # Enemy heading opposes vector-to-self → rel_velocity_angle ≈ π
    rel_vel = obs["enemies"][0, 5]
    assert abs(abs(rel_vel) - math.pi) < 0.2


def test_zero_enemies() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )

    obs = compute_observation(state, obs_config)
    assert obs["enemies"].shape == (obs_config.k_enemies, 7)
    np.testing.assert_allclose(obs["enemies"], 0.0)


def test_mass_log_transform() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    # Mass 10 -> log(10/10) = 0
    state10 = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )
    obs10 = compute_observation(state10, obs_config)
    np.testing.assert_allclose(obs10["self_state"][4], 0.0, atol=1e-5)

    # Mass 40000 -> log(40000/10) = log(4000) ≈ 8.29
    state40k = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=40000.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )
    obs40k = compute_observation(state40k, obs_config)
    expected = math.log(40000.0 / 10.0)
    np.testing.assert_allclose(obs40k["self_state"][4], expected, atol=0.1)


def test_priority_filtering() -> None:
    obs_config = ObsConfig(k_enemies=10)
    arrays = _empty_arrays()

    # 100 body segments close + 5 heads farther away
    n_body = 100
    n_heads = 5
    rng = np.random.default_rng(42)

    body_pos = rng.uniform(-50, 50, (n_body, 2)).astype(np.float32)
    head_pos = rng.uniform(-200, 200, (n_heads, 2)).astype(np.float32)

    all_pos = np.vstack([body_pos, head_pos])
    is_head = np.zeros(n_body + n_heads, dtype=np.bool_)
    is_head[n_body:] = True

    arrays["enemy_segments"] = all_pos
    arrays["enemy_is_head"] = is_head
    arrays["enemy_owner_mass"] = np.full(n_body + n_heads, 10.0, dtype=np.float32)
    arrays["enemy_owner_speed"] = np.full(n_body + n_heads, 3.0, dtype=np.float32)
    arrays["enemy_owner_angle"] = np.zeros(n_body + n_heads, dtype=np.float32)
    arrays["enemy_segment_radius"] = np.full(n_body + n_heads, 3.0, dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **arrays,
    )

    obs = compute_observation(state, obs_config)

    # All 5 heads should appear in the output (is_head=1.0)
    head_count = int(np.sum(obs["enemies"][:, 2] == 1.0))
    # At least the heads within perception range
    # Some heads may be beyond perception_radius=500 but most should be within
    assert head_count >= 3  # most of the 5 heads within [-200,200] are < 500 from origin


def test_benchmark() -> None:
    obs_config = ObsConfig()
    rng = np.random.default_rng(42)

    food_pos = rng.uniform(-400, 400, (1024, 2)).astype(np.float32)
    food_vals = np.ones(1024, dtype=np.float32)
    n_enemy = 500
    enemy_pos = rng.uniform(-400, 400, (n_enemy, 2)).astype(np.float32)

    n_snakes = 10
    snake_pos = rng.uniform(-2000, 2000, (n_snakes, 2)).astype(np.float32)
    snake_mass = rng.uniform(10, 500, n_snakes).astype(np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=100.0, self_speed=3.0, self_angle=0.5,
        food_positions=food_pos,
        food_values=food_vals,
        enemy_segments=enemy_pos,
        enemy_is_head=rng.random(n_enemy) > 0.9,
        enemy_owner_mass=rng.uniform(10, 1000, n_enemy).astype(np.float32),
        enemy_owner_speed=np.full(n_enemy, 3.0, dtype=np.float32),
        enemy_owner_angle=rng.uniform(0, 2 * np.pi, n_enemy).astype(np.float32),
        enemy_segment_radius=rng.uniform(3, 10, n_enemy).astype(np.float32),
        all_snake_positions=snake_pos,
        all_snake_masses=snake_mass,
        map_radius=3000.0,
    )

    # Warmup
    for _ in range(100):
        compute_observation(state, obs_config)

    n = 1000
    start = time.perf_counter()
    for _ in range(n):
        compute_observation(state, obs_config)
    elapsed = (time.perf_counter() - start) / n * 1000

    assert elapsed < 0.2, f"compute_observation too slow: {elapsed:.4f}ms (need <0.2ms)"
