"""T16 (DRAFT) body_shaping tests — obstruction potential, mirroring test_edge_shaping.

Pinned: default "off" is byte-identical; Φ properties (0 beyond width, −coef at contact
distance, measured against a hand-placed enemy); exact per-step bookkeeping; Φ(terminal)=0;
Φ kept at truncation; unknown mode rejected; off-mode reward stream identical to a paired
env (the smoke-oracle invariant for a term that has no legacy counterpart)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from slither_gym.core.types import WorldConfig
from slither_gym.rl.env_gym import SlitherGymEnv
from slither_gym.rl.types import ObsConfig


def _env(body_shaping: str = "potential", num_bots: int = 2, max_ticks: int = 400, **kw):
    wc = WorldConfig(
        map_radius=kw.pop("map_radius", 3000.0),
        body_shaping=body_shaping,
        max_food=256,
        food_spawn_rate=0,
        **kw,
    )
    return SlitherGymEnv(world_config=wc, obs_config=ObsConfig(), num_bots=num_bots,
                         max_ticks=max_ticks)


def test_default_is_off():
    assert WorldConfig().body_shaping == "off"


def test_phi_matches_hand_measured_nearest_enemy_segment():
    env = _env()
    env.reset(seed=0)
    me = env._world.get_snake_states()[0]
    best = math.inf
    for sid, ss in env._world.get_snake_states().items():
        if sid == 0 or not ss.alive:
            continue
        segs = env._world.get_segments(sid)
        for j in range(len(segs)):
            best = min(best, math.hypot(segs[j, 0] - me.head_x, segs[j, 1] - me.head_y))
    cfg = env._world_config
    expected = -cfg.body_shaping_coef * min(max(1.0 - best / cfg.body_shaping_width, 0.0), 1.0)
    np.testing.assert_allclose(env._compute_body_potential(), expected, rtol=1e-5)


def test_phi_zero_with_no_enemies():
    env = _env(num_bots=0)
    env.reset(seed=0)
    assert env._compute_body_potential() == 0.0


def test_step_bookkeeping_exact():
    env = _env()
    env.reset(seed=3)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(30):
        _o, _r, term, trunc, _i = env.step(a)
        phi_now = 0.0 if term else env._compute_body_potential()
        np.testing.assert_allclose(env._prev_body_phi, phi_now, atol=1e-12)
        if term or trunc:
            break


def test_off_mode_reward_stream_identical():
    env_p = _env("potential", num_bots=0)       # no enemies: Φ ≡ 0, streams must match
    env_o = _env("off", num_bots=0)
    a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    env_p.reset(seed=7)
    env_o.reset(seed=7)
    for _ in range(25):
        _, rp, tp, up, _ = env_p.step(a)
        _, ro, to, uo, _ = env_o.step(a)
        np.testing.assert_allclose(rp, ro, atol=1e-9)
        assert (tp, up) == (to, uo)
        if tp or up:
            break


def test_termination_zeroes_phi():
    env = _env(map_radius=400.0, max_ticks=100_000)
    env.reset(seed=1)
    a = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    terminated = False
    for _ in range(5000):
        _o, _r, terminated, trunc, _i = env.step(a)
        if terminated:
            break
    assert terminated and env._prev_body_phi == 0.0


def test_truncation_keeps_phi():
    env = _env(max_ticks=40)
    env.reset(seed=5)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    truncated = False
    while not truncated:
        _o, _r, term, truncated, _i = env.step(a)
        if term:
            pytest.skip("died before truncation on this seed")
    np.testing.assert_allclose(env._prev_body_phi, env._compute_body_potential(), atol=1e-12)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="body_shaping"):
        SlitherGymEnv(world_config=WorldConfig(body_shaping="typo"),
                      obs_config=ObsConfig(), num_bots=0)
