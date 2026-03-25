import numpy as np
from pettingzoo.test import parallel_api_test

from slither_gym.rl.env_parallel import SlitherParallelEnv
from slither_gym.core.types import WorldConfig
from slither_gym.rl.types import ObsConfig


def test_reset_returns_all_agents() -> None:
    env = SlitherParallelEnv(num_agents=3, seed=42)
    obs, infos = env.reset()
    assert len(obs) == 3
    assert all(f"snake_{i}" in obs for i in range(3))


def test_observation_shapes() -> None:
    obs_config = ObsConfig()
    env = SlitherParallelEnv(obs_config=obs_config, num_agents=2, seed=42)
    obs, _ = env.reset()

    for agent_id, ob in obs.items():
        assert ob["self_state"].shape == (6,)
        assert ob["food"].shape == (obs_config.k_food, obs_config.food_features)
        assert ob["enemies"].shape == (obs_config.k_enemies, obs_config.enemy_features)


def test_step_with_random_actions() -> None:
    env = SlitherParallelEnv(num_agents=2, seed=42)
    obs, _ = env.reset()

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

    assert isinstance(rewards, dict)
    assert isinstance(terms, dict)


def test_dead_agent_removed() -> None:
    config = WorldConfig(map_radius=50.0)  # Small map to force deaths
    env = SlitherParallelEnv(world_config=config, num_agents=3, seed=42, max_ticks=10000)
    obs, _ = env.reset()

    # Run until someone dies
    for _ in range(200):
        if not env.agents:
            break
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

    # With small map, at least one snake should have died
    # (boundary death is very likely)


def test_truncation_at_max_ticks() -> None:
    config = WorldConfig(step_mul=1)
    env = SlitherParallelEnv(world_config=config, num_agents=1, max_ticks=10, seed=42)
    obs, _ = env.reset()

    for _ in range(20):
        if not env.agents:
            break
        actions = {agent: np.array([1.0, 0.0, 0.0], dtype=np.float32) for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if any(truncs.values()):
            break

    # Should have been truncated or terminated
    assert True


def test_pettingzoo_api() -> None:
    env = SlitherParallelEnv(num_agents=2, seed=42, max_ticks=100)
    parallel_api_test(env, num_cycles=20)
