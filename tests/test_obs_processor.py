import math
import time

import numpy as np

from slither_gym.rl.obs_processor import compute_observation
from slither_gym.rl.types import EnemySnakeInfo, ObsConfig, RawGameState


def _empty_arrays() -> dict[str, np.ndarray]:
    return {
        "food_positions": np.zeros((0, 2), dtype=np.float32),
        "food_values": np.zeros(0, dtype=np.float32),
        "food_is_corpse": np.zeros(0, dtype=np.bool_),
        "own_segments": np.zeros((0, 2), dtype=np.float32),
        "enemy_segments": np.zeros((0, 2), dtype=np.float32),
        "enemy_is_head": np.zeros(0, dtype=np.bool_),
        "enemy_owner_mass": np.zeros(0, dtype=np.float32),
        "enemy_owner_speed": np.zeros(0, dtype=np.float32),
        "enemy_owner_angle": np.zeros(0, dtype=np.float32),
        "enemy_segment_radius": np.zeros(0, dtype=np.float32),
        "all_snake_positions": np.zeros((0, 2), dtype=np.float32),
        "all_snake_masses": np.zeros(0, dtype=np.float32),
    }


_SELF_DEFAULTS = {"self_segment_count": 10, "self_boosting": False}


def test_single_food() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()
    arrays["food_positions"] = np.array([[100.0, 0.0]], dtype=np.float32)
    arrays["food_values"] = np.array([1.0], dtype=np.float32)
    arrays["food_is_corpse"] = np.array([False], dtype=np.bool_)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
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
    arrays["food_is_corpse"] = np.array([False], dtype=np.bool_)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
    )

    obs = compute_observation(state, obs_config)
    np.testing.assert_allclose(obs["food"][0], [0.0, 0.0, 0.0], atol=1e-5)


def test_food_k_nearest() -> None:
    obs_config = ObsConfig(k_food=64)
    arrays = _empty_arrays()

    rng = np.random.default_rng(42)
    positions = rng.uniform(-400, 400, (100, 2)).astype(np.float32)
    arrays["food_positions"] = positions
    arrays["food_values"] = np.ones(100, dtype=np.float32)
    arrays["food_is_corpse"] = np.zeros(100, dtype=np.bool_)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
    )

    obs = compute_observation(state, obs_config)
    food = obs["food"]
    assert food.shape == (64, 3)

    nonzero_mask = np.any(food != 0, axis=1)
    nonzero = food[nonzero_mask]
    if len(nonzero) > 1:
        dists = np.sqrt(nonzero[:, 0] ** 2 + nonzero[:, 1] ** 2)
        assert np.all(dists[:-1] <= dists[1:] + 1e-5)


def test_zero_enemies() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
    )

    obs = compute_observation(state, obs_config)
    assert obs["enemies"].shape == (obs_config.k_enemies, obs_config.enemy_features)
    np.testing.assert_allclose(obs["enemies"], 0.0)
    assert obs["danger_segments"].shape == (obs_config.k_danger_segments, obs_config.danger_features)
    np.testing.assert_allclose(obs["danger_segments"], 0.0)


def test_mass_log_transform() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    state10 = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
    )
    obs10 = compute_observation(state10, obs_config)
    np.testing.assert_allclose(obs10["self_state"][4], 0.0, atol=1e-5)

    state40k = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=40000.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
    )
    obs40k = compute_observation(state40k, obs_config)
    expected = math.log(40000.0 / 10.0)
    np.testing.assert_allclose(obs40k["self_state"][4], expected, atol=0.1)


def _make_enemy_snake(
    snake_id: int, hx: float, hy: float,
    mass: float = 10.0, speed: float = 3.0, angle: float = 0.0,
    boosting: bool = False, n_segs: int = 20,
) -> EnemySnakeInfo:
    """Helper to create an EnemySnakeInfo with evenly-spaced segments behind head."""
    segs = np.zeros((n_segs, 2), dtype=np.float32)
    segs[0] = [hx, hy]
    dx = -math.cos(angle) * 5.0  # spacing behind head
    dy = -math.sin(angle) * 5.0
    for i in range(1, n_segs):
        segs[i] = [hx + dx * i, hy + dy * i]
    return EnemySnakeInfo(
        snake_id=snake_id, head_x=hx, head_y=hy,
        mass=mass, speed=speed, angle=angle, boosting=boosting,
        segments=segs,
    )


def test_enemy_slot_assignment() -> None:
    """Enemy snake is placed in the correct slot with is_active=1."""
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    snake = _make_enemy_snake(42, 100.0, 0.0)
    # Flat segments for danger_segments
    arrays["enemy_segments"] = snake.segments
    arrays["enemy_segment_radius"] = np.full(len(snake.segments), 3.0, dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
        enemy_snakes=(snake,),
    )

    mapping = {42: 5}  # assign to slot 5
    obs = compute_observation(state, obs_config, snake_slot_mapping=mapping)

    enemies = obs["enemies"]
    # Slot 5 should have data
    assert enemies[5, 31] == 1.0  # is_active
    np.testing.assert_allclose(enemies[5, 0], 100.0 / 500.0, atol=1e-5)  # head_dx
    np.testing.assert_allclose(enemies[5, 1], 0.0, atol=1e-5)  # head_dy
    # Other slots should be empty
    assert enemies[0, 31] == 0.0
    assert enemies[3, 31] == 0.0


def test_enemy_body_sampling() -> None:
    """12 body samples are evenly spaced along the snake."""
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    snake = _make_enemy_snake(1, 100.0, 0.0, n_segs=48)
    arrays["enemy_segments"] = snake.segments
    arrays["enemy_segment_radius"] = np.full(48, 3.0, dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
        enemy_snakes=(snake,),
    )

    mapping = {1: 0}
    obs = compute_observation(state, obs_config, snake_slot_mapping=mapping)

    # Body samples at indices 2-25 (12 x 2)
    body_data = obs["enemies"][0, 2:26].reshape(12, 2)
    # All should be nonzero (snake has 48 segments)
    assert np.any(body_data != 0)
    # First body sample should be close to head
    assert abs(body_data[0, 0] - 100.0 / 500.0) < 0.1


def test_enemy_metadata() -> None:
    """Enemy metadata fields (mass, speed, heading, boosting) are correct."""
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    snake = _make_enemy_snake(1, 100.0, 0.0, mass=100.0, speed=6.0,
                              angle=math.pi / 4, boosting=True)
    arrays["enemy_segments"] = snake.segments
    arrays["enemy_segment_radius"] = np.full(len(snake.segments), 3.0, dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
        enemy_snakes=(snake,),
    )

    mapping = {1: 0}
    obs = compute_observation(state, obs_config, snake_slot_mapping=mapping)
    e = obs["enemies"][0]

    np.testing.assert_allclose(e[26], math.log(100.0 / 10.0), atol=0.01)  # mass
    np.testing.assert_allclose(e[27], 6.0, atol=1e-5)  # speed
    np.testing.assert_allclose(e[28], math.cos(math.pi / 4), atol=1e-5)  # cos(angle)
    np.testing.assert_allclose(e[29], math.sin(math.pi / 4), atol=1e-5)  # sin(angle)
    assert e[30] == 1.0  # boosting
    assert e[31] == 1.0  # is_active


def test_enemy_no_mapping_fallback() -> None:
    """Without explicit mapping, enemies are assigned by distance."""
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    snake_near = _make_enemy_snake(1, 50.0, 0.0)
    snake_far = _make_enemy_snake(2, 300.0, 0.0)
    all_segs = np.concatenate([snake_near.segments, snake_far.segments])
    arrays["enemy_segments"] = all_segs
    arrays["enemy_segment_radius"] = np.full(len(all_segs), 3.0, dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
        enemy_snakes=(snake_near, snake_far),
    )

    obs = compute_observation(state, obs_config)  # no snake_slot_mapping
    # Nearest snake should be in slot 0
    np.testing.assert_allclose(obs["enemies"][0, 0], 50.0 / 500.0, atol=1e-5)
    assert obs["enemies"][0, 31] == 1.0


def test_danger_segments_shape_and_sorted() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    rng = np.random.default_rng(42)
    n = 200
    arrays["enemy_segments"] = rng.uniform(-300, 300, (n, 2)).astype(np.float32)
    arrays["enemy_is_head"] = np.zeros(n, dtype=np.bool_)
    arrays["enemy_owner_mass"] = np.full(n, 10.0, dtype=np.float32)
    arrays["enemy_owner_speed"] = np.full(n, 3.0, dtype=np.float32)
    arrays["enemy_owner_angle"] = np.zeros(n, dtype=np.float32)
    arrays["enemy_segment_radius"] = rng.uniform(3, 10, n).astype(np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
    )

    obs = compute_observation(state, obs_config)
    danger = obs["danger_segments"]
    assert danger.shape == (64, 3)

    # Non-zero entries should be sorted by distance
    nonzero_mask = np.any(danger != 0, axis=1)
    nonzero = danger[nonzero_mask]
    if len(nonzero) > 1:
        dists = np.sqrt(nonzero[:, 0] ** 2 + nonzero[:, 1] ** 2)
        assert np.all(dists[:-1] <= dists[1:] + 1e-5)

    # Third column is radius / 20
    assert np.all(danger[:, 2] >= 0)


def test_danger_segments_radius_normalization() -> None:
    obs_config = ObsConfig()
    arrays = _empty_arrays()

    arrays["enemy_segments"] = np.array([[100.0, 0.0]], dtype=np.float32)
    arrays["enemy_is_head"] = np.array([False], dtype=np.bool_)
    arrays["enemy_owner_mass"] = np.array([10.0], dtype=np.float32)
    arrays["enemy_owner_speed"] = np.array([3.0], dtype=np.float32)
    arrays["enemy_owner_angle"] = np.array([0.0], dtype=np.float32)
    arrays["enemy_segment_radius"] = np.array([10.0], dtype=np.float32)

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=10.0, self_speed=3.0, self_angle=0.0,
        map_radius=3000.0, **_SELF_DEFAULTS, **arrays,
    )

    obs = compute_observation(state, obs_config)
    np.testing.assert_allclose(obs["danger_segments"][0, 2], 10.0 / 20.0, atol=1e-5)


def test_benchmark() -> None:
    obs_config = ObsConfig()
    rng = np.random.default_rng(42)

    n_food = 1024
    food_pos = rng.uniform(-400, 400, (n_food, 2)).astype(np.float32)
    food_vals = np.ones(n_food, dtype=np.float32)

    # Build per-snake data
    enemy_snakes = []
    all_segs_list = []
    all_radius_list = []
    for i in range(10):
        hx, hy = float(rng.uniform(-300, 300)), float(rng.uniform(-300, 300))
        n_segs = int(rng.integers(10, 60))
        segs = np.column_stack([
            np.full(n_segs, hx) + rng.uniform(-50, 50, n_segs),
            np.full(n_segs, hy) + rng.uniform(-50, 50, n_segs),
        ]).astype(np.float32)
        enemy_snakes.append(EnemySnakeInfo(
            snake_id=i, head_x=hx, head_y=hy,
            mass=float(rng.uniform(10, 500)),
            speed=3.0, angle=float(rng.uniform(0, 2 * np.pi)),
            boosting=False, segments=segs,
        ))
        all_segs_list.append(segs)
        all_radius_list.append(rng.uniform(3, 10, n_segs).astype(np.float32))

    all_enemy_segs = np.concatenate(all_segs_list)
    all_radius = np.concatenate(all_radius_list)

    n_snakes = 10
    snake_pos = rng.uniform(-2000, 2000, (n_snakes, 2)).astype(np.float32)
    snake_mass = rng.uniform(10, 500, n_snakes).astype(np.float32)

    mapping = {s.snake_id: i for i, s in enumerate(enemy_snakes[:16])}

    state = RawGameState(
        self_x=0.0, self_y=0.0, self_mass=100.0, self_speed=3.0, self_angle=0.5,
        self_segment_count=50, self_boosting=False,
        food_positions=food_pos,
        food_values=food_vals,
        food_is_corpse=rng.random(n_food) > 0.9,
        own_segments=rng.uniform(-100, 100, (50, 2)).astype(np.float32),
        enemy_segments=all_enemy_segs,
        enemy_is_head=np.zeros(len(all_enemy_segs), dtype=np.bool_),
        enemy_owner_mass=np.full(len(all_enemy_segs), 10.0, dtype=np.float32),
        enemy_owner_speed=np.full(len(all_enemy_segs), 3.0, dtype=np.float32),
        enemy_owner_angle=np.zeros(len(all_enemy_segs), dtype=np.float32),
        enemy_segment_radius=all_radius,
        all_snake_positions=snake_pos,
        all_snake_masses=snake_mass,
        map_radius=3000.0,
        enemy_snakes=tuple(enemy_snakes),
    )

    # Warmup
    for _ in range(100):
        compute_observation(state, obs_config, snake_slot_mapping=mapping)

    n = 1000
    start = time.perf_counter()
    for _ in range(n):
        compute_observation(state, obs_config, snake_slot_mapping=mapping)
    elapsed = (time.perf_counter() - start) / n * 1000

    assert elapsed < 0.5, f"compute_observation too slow: {elapsed:.4f}ms (need <0.5ms)"
