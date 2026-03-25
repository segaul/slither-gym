import math

import numpy as np

from slither_gym.rl.bot_policy import BotPolicy
from slither_gym.core.types import WorldConfig
from slither_gym.rl.types import ObsConfig


def _make_obs(
    food: np.ndarray | None = None,
    enemies: np.ndarray | None = None,
    heading_angle: float = 0.0,
) -> dict[str, np.ndarray]:
    obs_config = ObsConfig()
    if food is None:
        food = np.zeros((obs_config.k_food, obs_config.food_features), dtype=np.float32)
    if enemies is None:
        enemies = np.zeros((obs_config.k_enemies, obs_config.enemy_features), dtype=np.float32)

    self_state = np.array([
        0.0, 0.0,
        math.cos(heading_angle), math.sin(heading_angle),
        0.0, 3.0,
    ], dtype=np.float32)

    return {"self_state": self_state, "food": food, "enemies": enemies}


def test_seek_food() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    bot = BotPolicy(config, rng)

    obs_config = ObsConfig()
    food = np.zeros((obs_config.k_food, obs_config.food_features), dtype=np.float32)
    food[0] = [0.5, 0.0, 1.0]  # food directly ahead (normalized)

    obs = _make_obs(food=food)
    action = bot.act(obs)

    # Should move roughly toward food (cos > 0)
    assert action[0] > 0.5


def test_flee_danger() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    bot = BotPolicy(config, rng)

    obs_config = ObsConfig()
    enemies = np.zeros((obs_config.k_enemies, obs_config.enemy_features), dtype=np.float32)
    # Enemy head very close, directly ahead
    enemies[0, 0] = 0.1   # rel_x
    enemies[0, 1] = 0.0   # rel_y
    enemies[0, 2] = 1.0   # is_head
    enemies[0, 3] = 1.0   # mass
    enemies[0, 4] = 3.0   # speed

    obs = _make_obs(enemies=enemies)
    action = bot.act(obs)

    # Should flee (move away, cos < 0)
    assert action[0] < 0.0


def test_no_boost() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    bot = BotPolicy(config, rng)

    obs = _make_obs()
    for _ in range(100):
        action = bot.act(obs)
        assert action[2] <= 0.5


def test_noise_variance() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    bot = BotPolicy(config, rng)

    obs_config = ObsConfig()
    food = np.zeros((obs_config.k_food, obs_config.food_features), dtype=np.float32)
    food[0] = [1.0, 0.0, 1.0]  # food directly ahead

    obs = _make_obs(food=food)

    angles = []
    for _ in range(100):
        action = bot.act(obs)
        angle = math.atan2(float(action[1]), float(action[0]))
        angles.append(angle)

    # Should not be identical (noise)
    std = np.std(angles)
    assert std > 0.01, f"Noise too low: std={std}"
    assert std < 1.0, f"Noise too high: std={std}"


def test_random_walk() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(42)
    bot = BotPolicy(config, rng)

    # No food, no enemies
    obs = _make_obs(heading_angle=0.5)
    action = bot.act(obs)

    # Should output something near current heading
    output_angle = math.atan2(float(action[1]), float(action[0]))
    assert abs(output_angle - 0.5) < 1.0  # within noise margin
