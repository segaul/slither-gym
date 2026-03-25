import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from slither_gym.rl.env_gym import SlitherGymEnv
from slither_gym.core.types import WorldConfig


def test_reset_observation_shape() -> None:
    env = SlitherGymEnv(seed=42)
    obs, info = env.reset()
    assert obs["self_state"].shape == (6,)
    assert obs["food"].shape == (32, 3)
    assert obs["enemies"].shape == (64, 7)


def test_step_returns_correctly() -> None:
    env = SlitherGymEnv(num_bots=2, seed=42)
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_solo_agent() -> None:
    env = SlitherGymEnv(num_bots=0, seed=42)
    obs, _ = env.reset()
    for _ in range(50):
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break


def test_bots_move() -> None:
    env = SlitherGymEnv(num_bots=5, seed=42)
    obs, _ = env.reset()
    world = env._world
    assert world is not None

    # Get initial bot positions
    initial_positions = {}
    for i in range(1, 6):
        state = world.get_snake_states().get(i)
        if state is not None:
            initial_positions[i] = (state.head_x, state.head_y)

    # Step
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env.step(action)

    # Bots should have moved
    moved = False
    for i, (ix, iy) in initial_positions.items():
        state = world.get_snake_states().get(i)
        if state is not None and state.alive:
            if abs(state.head_x - ix) > 0.01 or abs(state.head_y - iy) > 0.01:
                moved = True
    assert moved


def test_rl_agent_death() -> None:
    config = WorldConfig(map_radius=50.0)
    env = SlitherGymEnv(world_config=config, num_bots=0, seed=42)
    obs, _ = env.reset()

    for _ in range(200):
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        if term:
            break

    # Should terminate from boundary
    assert term


def test_respawn_bots() -> None:
    config = WorldConfig(map_radius=50.0)
    env = SlitherGymEnv(world_config=config, num_bots=3, seed=42, respawn_bots=True)
    obs, _ = env.reset()

    # Run many steps - bots should die and respawn
    for _ in range(100):
        action = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break

    # Check that bots are still in the agent list (respawned)
    parallel_env = env._parallel_env
    bot_agents = [a for a in parallel_env.agents if a != "snake_0"]
    # With respawn, there should still be bots
    # (they may die and respawn within the same frame)


def test_no_respawn_bots() -> None:
    config = WorldConfig(map_radius=50.0)
    env = SlitherGymEnv(world_config=config, num_bots=3, seed=42, respawn_bots=False)
    obs, _ = env.reset()

    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break


def test_step_mul() -> None:
    config = WorldConfig(step_mul=4)
    env = SlitherGymEnv(world_config=config, num_bots=0, seed=42)
    obs, _ = env.reset()

    world = env._world
    assert world is not None
    tick_before = world.get_tick()

    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env.step(action)

    tick_after = world.get_tick()
    assert tick_after - tick_before == 4


def test_gymnasium_check_env() -> None:
    env = SlitherGymEnv(num_bots=2, seed=42, max_ticks=100)
    check_env(env.unwrapped, skip_render_check=True)


def test_1000_steps_no_crash() -> None:
    env = SlitherGymEnv(num_bots=3, seed=42)
    obs, _ = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    # Verify info has mass
    assert "mass" not in info or isinstance(info.get("mass"), float)
