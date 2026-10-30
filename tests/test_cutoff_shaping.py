"""T35 cut-off hazard potential: geometry unit tests + knob-off byte-identity +
telescoping golden over a fixed rollout."""
from __future__ import annotations

import math

import numpy as np
import pytest

from slither_gym.rl.env_gym import SlitherGymEnv
from slither_gym.core.types import WorldConfig


def _env(**over):
    wc = WorldConfig(map_radius=1500.0, max_food=500, **over)
    return SlitherGymEnv(world_config=wc, num_bots=3, max_ticks=300, seed=11)


def test_knob_off_byte_identity():
    a, b = _env(), _env(cutoff_shaping="off")
    oa, _ = a.reset(seed=5)
    ob, _ = b.reset(seed=5)
    act = np.array([1.0, 0.0, 0.0], np.float32)
    for _ in range(40):
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
        assert ra == rb and ta == tb and tra == trb
        for k in oa:
            assert np.array_equal(oa[k], ob[k])
        if ta or tra:
            break


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        _env(cutoff_shaping="on")


def test_potential_geometry():
    env = _env(cutoff_shaping="potential", cutoff_shaping_coef=2.0,
               cutoff_shaping_width=600.0)
    env.reset(seed=5)
    me = env._world.get_snake_states()[0]

    class S:  # minimal stand-in with the fields the potential reads
        def __init__(self, x, y, ang, boosting, alive=True):
            self.head_x, self.head_y, self.angle = x, y, ang
            self.boosting, self.alive = boosting, alive

    ahead_x = me.head_x + 300 * math.cos(me.angle)
    ahead_y = me.head_y + 300 * math.sin(me.angle)
    closing_ang = me.angle + math.pi  # pointing straight back at us
    real_states = env._world.get_snake_states

    def fake(states):
        env._world.get_snake_states = lambda: states
        try:
            return env._compute_cutoff_potential()
        finally:
            env._world.get_snake_states = real_states

    base = {0: me}
    assert fake({**base, 1: S(ahead_x, ahead_y, closing_ang, True)}) == pytest.approx(
        -2.0 * (1 - 300 / 600), abs=1e-6)          # boosting+ahead+closing -> priced
    assert fake({**base, 1: S(ahead_x, ahead_y, closing_ang, False)}) == 0.0  # not boosting
    assert fake({**base, 1: S(ahead_x, ahead_y, me.angle, True)}) == 0.0      # fleeing, not closing
    behind_x = me.head_x - 300 * math.cos(me.angle)
    behind_y = me.head_y - 300 * math.sin(me.angle)
    assert fake({**base, 1: S(behind_x, behind_y, closing_ang, True)}) == 0.0  # behind us
    far_x = me.head_x + 900 * math.cos(me.angle)
    assert fake({**base, 1: S(far_x, me.head_y, closing_ang, True)}) == 0.0    # out of width


def test_telescoping_golden():
    """Sum of shaping rewards over an episode == gamma-weighted telescope of Φ:
    with Φ(terminal)=0, sum(γΦ' − Φ) = −Φ(s0) + (γ−1)·ΣΦ(s_t). We verify by
    replaying the identical seed with the knob off and differencing rewards."""
    on = _env(cutoff_shaping="potential", cutoff_shaping_coef=2.0)
    off = _env()
    on.reset(seed=9)
    off.reset(seed=9)
    act = np.array([0.8, 0.6, 0.0], np.float32)
    phis = [on._prev_cutoff_phi]
    diff_sum = 0.0
    for _ in range(60):
        _, r_on, t1, tr1, _ = on.step(act)
        _, r_off, t2, tr2, _ = off.step(act)
        assert (t1, tr1) == (t2, tr2)  # shaping must not change dynamics
        diff_sum += r_on - r_off
        phis.append(on._prev_cutoff_phi)
        if t1 or tr1:
            break
    g = 0.997
    expected = sum(g * phis[i + 1] - phis[i] for i in range(len(phis) - 1))
    assert diff_sum == pytest.approx(expected, abs=1e-5)
