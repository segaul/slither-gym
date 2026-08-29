"""S5: the opt-in ``obs_schema='v5'`` env mode (env_gym + env_parallel).

Covers: declared observation_space == built obs (shapes/dtypes/containment),
last-action slots carry the previous COMMANDED action (pre-delay-queue —
see env_obs_v5's module docstring), correct interplay with the S4 delay
queue, and the V4 default path staying byte-identical.
"""

from __future__ import annotations

import math

import numpy as np

from slither_gym.core.types import WorldConfig
from slither_gym.obs.schema_v5 import ObsConfigV5
from slither_gym.rl.env_gym import SlitherGymEnv
from slither_gym.rl.env_parallel import SlitherParallelEnv

CFG = ObsConfigV5()
V5_SHAPES = {
    "self_state": (12,),
    "food": (CFG.k_food, 3),
    "enemies": (CFG.k_enemies, CFG.enemy_features),
    "danger_segments": (CFG.k_danger, 3),
    "own_body": (CFG.k_own_body, 2),
}


def _assert_v5_obs(obs: dict, space) -> None:
    assert set(obs) == set(V5_SHAPES)
    for k, arr in obs.items():
        assert arr.shape == V5_SHAPES[k], k
        assert arr.dtype == np.float32, k
        assert np.all(np.isfinite(arr)), k
    assert space.contains(obs)


def test_env_gym_v5_space_and_obs() -> None:
    env = SlitherGymEnv(
        world_config=WorldConfig(),
        num_bots=3,
        max_ticks=200,
        seed=7,
        obs_schema="v5",
    )
    obs, _ = env.reset(seed=7)
    _assert_v5_obs(obs, env.observation_space)

    # reset: last-action slots seeded with spawn heading / no boost
    angle = env._world.get_snake_states()[0].angle
    np.testing.assert_allclose(
        obs["self_state"][9:12],
        np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32),
        rtol=1e-6,
    )
    assert obs["self_state"][7] == 0.0  # no boost commanded yet

    for _ in range(5):
        obs, _r, term, trunc, _info = env.step(
            np.array([0.6, 0.8, 1.0], dtype=np.float32)
        )
        if term or trunc:
            break
        _assert_v5_obs(obs, env.observation_space)
        # the action was normalized to unit length (0.6, 0.8) and echoed back
        np.testing.assert_allclose(
            obs["self_state"][9:12], [0.6, 0.8, 1.0], rtol=1e-6
        )
        assert obs["self_state"][7] == 1.0  # commanded boost echoed


def test_env_gym_v5_commanded_not_applied_under_delay() -> None:
    """With an action delay LONGER than one RL step (8 ticks > step_mul 4),
    the physics has not applied the boost yet after one step — but the obs
    must already carry the COMMANDED boost (deployment knows only what it
    sent)."""
    env = SlitherGymEnv(
        world_config=WorldConfig(action_delay_ticks=8),
        num_bots=0,
        max_ticks=200,
        seed=3,
        obs_schema="v5",
    )
    env.reset(seed=3)
    obs, _r, term, _trunc, _info = env.step(
        np.array([1.0, 0.0, 1.0], dtype=np.float32)
    )
    assert not term
    st = env._world.get_snake_states()[0]
    assert not st.boosting  # queue still holds the seeded no-boost actions
    assert obs["self_state"][7] == 1.0  # ... but we COMMANDED boost
    assert obs["self_state"][11] == 1.0


def test_env_parallel_v5() -> None:
    env = SlitherParallelEnv(
        world_config=WorldConfig(),
        num_agents=3,
        max_ticks=200,
        seed=11,
        obs_schema="v5",
    )
    obs, _ = env.reset(seed=11)
    assert set(obs) == set(env.agents)
    for agent_id, o in obs.items():
        _assert_v5_obs(o, env.observation_space(agent_id))

    actions = {
        a: np.array([0.0, 1.0, 1.0], dtype=np.float32) for a in env.agents
    }
    obs, _rw, terms, _tr, _inf = env.step(actions)
    for agent_id in env.agents:
        if terms.get(agent_id):
            continue
        o = obs[agent_id]
        _assert_v5_obs(o, env.observation_space(agent_id))
        np.testing.assert_allclose(
            o["self_state"][9:12], [0.0, 1.0, 1.0], rtol=1e-6
        )


def test_v4_default_untouched() -> None:
    """The default obs_schema stays 'v4' and produces bit-identical obs to a
    pre-S5 construction (no new kwargs)."""
    a = SlitherGymEnv(num_bots=2, max_ticks=100, seed=5)
    b = SlitherGymEnv(num_bots=2, max_ticks=100, seed=5, obs_schema="v4")
    oa, _ = a.reset(seed=5)
    ob, _ = b.reset(seed=5)
    assert set(oa) == set(ob)
    assert "minimap" in oa  # V4 keys, not V5
    for k in oa:
        assert oa[k].tobytes() == ob[k].tobytes(), k
    act = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    for _ in range(3):
        oa = a.step(act)[0]
        ob = b.step(act)[0]
        for k in oa:
            assert oa[k].tobytes() == ob[k].tobytes(), k
