"""P0.5 (frozen_eval_v8): opponent size-distribution law tests.

bot_sct_law="lognormal" spawns/respawns bots with sct drawn from the measured
real-opponent size distribution (p10 2, p50 22, p90 178, max 262 — 8-game set,
docs/SIM_REALISM_STATE.md in slither-rl). Default (None) must stay byte-identical
to the legacy fixed-size spawn.
"""
from __future__ import annotations

import numpy as np

from slither_gym.rl.env_gym import SlitherGymEnv


def _bot_scts(env: SlitherGymEnv) -> list[int]:
    states = env._world.get_snake_states()  # type: ignore[union-attr]
    return [s.segment_count for i, s in states.items() if i != 0 and s.alive]


def test_default_is_legacy_fixed_size() -> None:
    env = SlitherGymEnv(num_bots=8, seed=3, max_ticks=100)
    env.reset()
    scts = _bot_scts(env)
    assert len(scts) == 8
    # Legacy: every snake spawns at initial_mass -> initial_segments.
    assert all(s == env._world_config.initial_segments for s in scts)


def test_default_rng_stream_untouched() -> None:
    """bot_sct_law=None must draw zero RNG -> identical obs to an unmodified env."""
    a = SlitherGymEnv(num_bots=4, seed=7, max_ticks=100)
    b = SlitherGymEnv(num_bots=4, seed=7, max_ticks=100, bot_sct_law=None)
    oa, _ = a.reset()
    ob, _ = b.reset()
    for k in oa:
        np.testing.assert_array_equal(oa[k], ob[k])


def test_lognormal_sizes_span_and_clip() -> None:
    env = SlitherGymEnv(num_bots=32, seed=11, max_ticks=100, bot_sct_law="lognormal")
    scts: list[int] = []
    for seed in range(11, 19):
        env.reset(seed=seed)
        scts.extend(_bot_scts(env))
    arr = np.array(scts)
    cap = env._world_config.max_segments_per_snake
    floor = env._world_config.initial_segments
    assert arr.min() >= floor  # spawn floor: mass clamps at initial_mass
    assert arr.max() <= cap
    # Distribution actually spans sizes: median near 22 (broadly), some big ones.
    assert np.median(arr) < 60
    assert arr.max() > 100


def test_lognormal_deterministic_per_seed() -> None:
    e1 = SlitherGymEnv(num_bots=16, seed=5, max_ticks=100, bot_sct_law="lognormal")
    e2 = SlitherGymEnv(num_bots=16, seed=5, max_ticks=100, bot_sct_law="lognormal")
    e1.reset()
    e2.reset()
    assert _bot_scts(e1) == _bot_scts(e2)


def test_respawn_redraws_size() -> None:
    env = SlitherGymEnv(num_bots=12, seed=13, max_ticks=4000, bot_sct_law="lognormal")
    env.reset()
    sizes_seen: set[int] = set(_bot_scts(env))
    for _ in range(300):
        _, _, term, trunc, _ = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        sizes_seen.update(_bot_scts(env))
        if term or trunc:
            break
    # Respawned bots draw fresh sizes -> more distinct sizes than one reset gave.
    assert len(sizes_seen) >= 8
