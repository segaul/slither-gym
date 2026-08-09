"""Pin the scripted-bot partial-obs fast path (env-throughput batch).

_compute_bot_observation_fast must produce BITWISE the same values as the
full compute_observation path for every field BotPolicy consumes:
  self_state[0:8], the whole food channel, and enemies columns
  {0, 1, 26, 27, 28, 29, 30, 31}. Everything else is contractually zero.
"""
import numpy as np

from slither_gym.core.types import WorldConfig
from slither_gym.rl.env_gym import SlitherGymEnv
from slither_gym.rl.obs_processor import compute_observation

CONSUMED_ENEMY_COLS = [0, 1, 26, 27, 28, 29, 30, 31]
BODY_COLS = list(range(2, 26))


def _env(**kw) -> SlitherGymEnv:
    return SlitherGymEnv(
        world_config=WorldConfig(max_snakes=12),
        num_bots=8,
        max_ticks=500,
        seed=7,
        **kw,
    )


def _assert_fast_matches_full(env: SlitherGymEnv) -> int:
    states = env._world.get_snake_states()
    food_pos = env._world.get_food_positions()
    food_vals = env._world.get_food_values()
    food_corpse = env._world.get_food_is_corpse()
    checked = 0
    for i in range(1, 1 + env._num_bots):
        st = states.get(i)
        if st is None or not st.alive:
            continue
        fast = env._compute_bot_observation_fast(i, st, states)
        raw = env._build_raw_state(i, st, states, food_pos, food_vals, food_corpse)
        full = compute_observation(raw, env._obs_config)

        assert np.array_equal(fast["self_state"][:8], full["self_state"][:8])
        assert np.all(fast["self_state"][8:] == 0.0)
        assert np.array_equal(fast["food"], full["food"])
        assert np.array_equal(
            fast["enemies"][:, CONSUMED_ENEMY_COLS],
            full["enemies"][:, CONSUMED_ENEMY_COLS],
        )
        assert np.all(fast["enemies"][:, BODY_COLS] == 0.0)
        for k in ("prey", "danger_segments", "own_body", "minimap"):
            assert fast[k].shape == full[k].shape
            assert np.all(fast[k] == 0.0)
        checked += 1
    return checked


def test_fast_bot_obs_bitwise_on_consumed_fields() -> None:
    env = _env()
    env.reset(seed=7)
    total = _assert_fast_matches_full(env)
    rng = np.random.default_rng(0)
    for t in range(120):
        a = rng.uniform(-1, 1, 3).astype(np.float32)
        a[2] = 1.0 if rng.uniform() < 0.2 else 0.0
        _obs, _r, term, trunc, _info = env.step(a)
        if term or trunc:
            env.reset(seed=100 + t)
        if t % 20 == 0:
            total += _assert_fast_matches_full(env)
    assert total > 20, f"too few live-bot comparisons exercised ({total})"


def test_fast_bot_obs_bitwise_with_size_law() -> None:
    env = _env(
        bot_sct_law="lognormal",
        bot_sct_log_median=22.0,
        bot_sct_log_sigma=1.632,
    )
    env.reset(seed=11)
    rng = np.random.default_rng(1)
    total = _assert_fast_matches_full(env)
    for t in range(60):
        a = rng.uniform(-1, 1, 3).astype(np.float32)
        _obs, _r, term, trunc, _info = env.step(a)
        if term or trunc:
            env.reset(seed=200 + t)
        if t % 15 == 0:
            total += _assert_fast_matches_full(env)
    assert total > 10


def test_scripted_bots_use_partial_cache_and_policy_bots_full() -> None:
    env = _env()
    env.reset(seed=7)
    assert env._bot_obs_partial, "scripted bots should take the fast path"
    assert set(env._bot_obs_cache) == env._bot_obs_partial

    class _FullObsPolicy:
        def act(self, obs, **kw):
            # A neural-style policy that reads the whole obs.
            assert obs["minimap"].shape[0] > 0
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Mid-episode swap: the partial entry must be upgraded to a full obs.
    env.set_bot_policies({1: _FullObsPolicy()})
    if 1 in env._bot_obs_cache:
        assert 1 not in env._bot_obs_partial
    # And after the next cache refresh, bot 1 keeps getting the full build.
    env.reset(seed=8)
    assert 1 not in env._bot_obs_partial
    assert any(i in env._bot_obs_partial for i in range(2, 9))
