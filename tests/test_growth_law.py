"""R3 acceptance tests: the real growth law (pellet value -> mass -> sct/fam).

Three jobs:
  1. Legacy byte-identity — growth_law="legacy" (the default) must leave every
     pre-R3 trajectory bit-exact (golden-hash test recorded from the pre-R3
     code at commit 4663f57).
  2. The law itself — the fpsls/fmlts inverse is exact, and hand-checked
     points reproduce the client LUTs and the real measurements they were
     never fitted to (M1: the human's sct 43 -> length 711).
  3. The R3 fix — pellet-by-pellet growth through a real World follows the
     client trajectory, and time-to-sct-256 at measured food density lands in
     MINUTES (real) vs ~23 s (legacy), the diagnosed E32 degenerate optimum.
"""
import dataclasses
import hashlib
import math

import numpy as np
import pytest

from slither_gym.core.growth import (
    MIN_REAL_SCT,
    build_mass_luts,
    real_mass,
    real_mass_continuous,
    sct_fam_from_mass,
)
from slither_gym.core.realism import TICK_HZ, realistic_world_config
from slither_gym.core.snake import _expected_segments, initial_segment_count
from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World
from slither_gym.obs.visibility import visible_state_from_world


# --------------------------------------------------------------------------
# 1. Legacy byte-identity
# --------------------------------------------------------------------------

def _trajectory_hash(config: WorldConfig, seed: int) -> str:
    """400-tick two-snake trajectory digest (mass, pose, sct, radius, turn,
    food). Any behavioral drift in the legacy path changes it."""
    w = World(config, seed=seed)
    w.spawn_snake(0)
    w.spawn_snake(1, mass=55.0)
    h = hashlib.sha256()
    for t in range(400):
        a = 0.13 * t
        w.step({0: (math.cos(a), math.sin(a), t % 11 == 0),
                1: (math.cos(-a), math.sin(-a), False)})
        for sid in (0, 1):
            s = w.get_snake_states()[sid]
            h.update(np.float64([s.mass, s.head_x, s.head_y, s.angle,
                                 s.segment_count, s.segment_radius,
                                 s.turn_rate]).tobytes())
        h.update(w.get_food_positions().tobytes())
        h.update(w.get_food_values().tobytes())
    return h.hexdigest()


# Recorded from the PRE-R3 code (commit 4663f57) with the identical harness.
_GOLDEN = {
    0: "42dad6aadd7017604c86179ec070cc2e39d601f14ec2cf5bba8104ddb27dc48c",
    7: "3ebdabefaf81918be11688b4f9632485e253ecc70e2fdd0e2d429a6cb2c686a1",
}


@pytest.mark.parametrize("seed", sorted(_GOLDEN))
def test_legacy_default_is_byte_identical_to_pre_r3(seed: int) -> None:
    assert WorldConfig().growth_law == "legacy"
    assert _trajectory_hash(WorldConfig(), seed) == _GOLDEN[seed]


def test_explicit_legacy_equals_default() -> None:
    explicit = dataclasses.replace(WorldConfig(), growth_law="legacy")
    assert _trajectory_hash(explicit, 0) == _GOLDEN[0]


def test_unknown_growth_law_fails_loudly() -> None:
    c = dataclasses.replace(WorldConfig(), growth_law="nope")
    with pytest.raises(ValueError):
        _expected_segments(10.0, c)


# --------------------------------------------------------------------------
# 2. The law: hand checks and the exact inverse
# --------------------------------------------------------------------------

def test_lut_hand_checked_against_the_client_recurrence() -> None:
    """First entries computed by hand from the client's own construction
    (fmlts[b] = (1 - b/430)^2.25; fpsls[b] = fpsls[b-1] + 1/fmlts[b-1])."""
    fpsls, fmlts = build_mass_luts()
    assert fmlts[0] == 1.0
    assert fmlts[1] == pytest.approx((1.0 - 1.0 / 430.0) ** 2.25, abs=0)
    assert fpsls[0] == 0.0
    assert fpsls[1] == 1.0                       # + 1/fmlts[0]
    assert fpsls[2] == pytest.approx(1.0 + (1.0 - 1.0 / 430.0) ** -2.25)


def test_real_mass_reproduces_the_m1_measurement() -> None:
    """M1 (docs/REAL_GAME_DATA.md): the human at sct 43 measured length 711.
    The law was NOT fitted to it — mscps=430 comes from the bitwise LUT
    verification — so this is an independent check (0.4% off)."""
    assert real_mass(43, 0.5) == pytest.approx(711.0, rel=0.005)
    # Spawn: real snakes spawn at sct 2, mass ~10 — the sim's initial_mass.
    assert real_mass(2, 0.0) == pytest.approx(10.079, abs=0.001)


def test_marginal_segment_cost_is_superlinear_and_over_16_5() -> None:
    """The R3 brief's anchor: >= 16.5 mass/segment at the relevant sizes
    (~1.3 mass per pellet -> ~13+ pellets per segment, vs legacy's 6.25
    SEGMENTS per pellet: the ~79x)."""
    costs = {s: real_mass(s + 1, 0.0) - real_mass(s, 0.0) for s in (22, 43, 77, 178, 255)}
    assert all(c >= 16.5 for c in costs.values())
    assert costs[255] > costs[77] > costs[22]  # superlinear
    # Segments per mean pellet at the median real size (sct 22): ~0.077,
    # i.e. the legacy 6.25 seg/pellet was ~79-81x too fast.
    seg_per_pellet = 6.25 * 0.208 / costs[22]
    assert 6.25 / seg_per_pellet == pytest.approx(81.0, abs=3.0)


@pytest.mark.parametrize("sct,fam", [(2, 0.0), (5, 0.3), (22, 0.9619),
                                     (43, 0.5), (256, 0.9), (429, 0.0)])
def test_sct_fam_inverse_is_exact(sct: int, fam: float) -> None:
    got_sct, got_fam = sct_fam_from_mass(real_mass(sct, fam))
    assert got_sct == sct
    assert got_fam == pytest.approx(fam, abs=1e-9)


def test_inverse_floors_at_the_client_minimum() -> None:
    assert sct_fam_from_mass(0.0) == (MIN_REAL_SCT, 0.0)
    assert sct_fam_from_mass(10.0) == (MIN_REAL_SCT, 0.0)  # initial_mass


# --------------------------------------------------------------------------
# 3. The R3 fix in the sim
# --------------------------------------------------------------------------

def _quiet_real_world(**overrides: object) -> World:
    """Realistic preset, but no ambient food and no bots — a clean bench."""
    c = realistic_world_config(**overrides)
    c = dataclasses.replace(c, food_density_per_1e6=0.0)
    return World(c, seed=3)


def test_real_spawn_is_sct_2() -> None:
    c = realistic_world_config()
    assert c.growth_law == "real"
    assert initial_segment_count(c) == 2
    w = _quiet_real_world()
    w.spawn_snake(0)
    assert w.get_snake_states()[0].segment_count == 2
    assert len(w.get_segments(0)) == 2


def test_pellet_by_pellet_growth_matches_the_client_trajectory() -> None:
    """Feed one mean pellet (value 6.25) per step and check the sim walks the
    client's own curve: mass = 10 + 1.3n, and sct flips EXACTLY when mass
    crosses real_mass(sct+1, 0) — never on the legacy 1-per-mass schedule."""
    w = _quiet_real_world()
    c = w.get_config()
    w.spawn_snake(0)
    st = w.get_snake_states()[0]
    n_eaten = 0
    for _ in range(600):
        # Steer toward the centre so the bench never touches the border.
        r = math.hypot(st.head_x, st.head_y)
        cos_a, sin_a = (-st.head_x / r, -st.head_y / r) if r > 1.0 else (1.0, 0.0)
        # Drop the next pellet right on the head: eaten this very tick.
        w._food.spawn_at(st.head_x, st.head_y, 6.25)
        res = w.step({0: (cos_a, sin_a, False)})
        assert res[0].alive
        if res[0].mass_delta > 0.0:
            n_eaten += 1
            assert res[0].mass_delta == pytest.approx(6.25 * c.pellet_mass_per_value)
        st = w.get_snake_states()[0]
        expected_mass = c.initial_mass + n_eaten * 6.25 * c.pellet_mass_per_value
        assert st.mass == pytest.approx(expected_mass)
        # sct follows the LUT inverse of mass exactly (10.0 floors at sct 2),
        # up to the one-tick chain lag: World.step eats AFTER move(), so the
        # segment appears on the next tick's move.
        assert 0 <= sct_fam_from_mass(st.mass)[0] - st.segment_count <= 1
    assert n_eaten >= 590
    w.step({0: (1.0, 0.0, False)})  # let the chain catch up one tick
    st = w.get_snake_states()[0]
    assert st.segment_count == sct_fam_from_mass(st.mass)[0]
    # ~600 pellets = +780 mass -> mass ~790 -> sct 46 by the client law.
    # (Legacy would have pinned the 256 cap ~2600 mass ago.)
    assert st.segment_count == sct_fam_from_mass(10.0 + n_eaten * 1.3)[0]
    assert 40 <= st.segment_count <= 55


def test_time_to_sct_256_cap_minutes_not_seconds() -> None:
    """The R3 headline. Pellet encounter rate at measured constants
    (2 * collect_radius * speed * density) times the value->mass conversion:
    legacy saturates the physics cap in ~23 s of a 200 s episode; the real
    law takes ~80 min of pure foraging — the cap stops being reachable-then-
    inert and mass stays physically meaningful all episode."""
    c = realistic_world_config()
    pellets_per_s = (
        2.0 * c.collect_radius_base * c.base_speed * TICK_HZ
        * 62.0e-6  # measured density (the preset's food_density_per_1e6/1e6)
    )
    assert pellets_per_s == pytest.approx(1.688, abs=0.01)
    mean_value = 6.25
    legacy_s = (256.0 - 10.0) / (pellets_per_s * mean_value)
    assert legacy_s == pytest.approx(23.3, abs=0.5)  # the E32 saturation time
    real_s = (real_mass(256, 0.0) - c.initial_mass) / (
        pellets_per_s * mean_value * c.pellet_mass_per_value
    )
    assert real_s > 45 * 60  # ~4900 s ~= 82 min: minutes-to-hours, not seconds
    assert real_s / legacy_s > 75  # the ~79x, compounded by superlinearity


def test_mass_stays_rewardable_past_the_segment_cap() -> None:
    """Reward = mass_delta on the REAL mass scale: even at the sct-256
    physics cap the currency keeps moving (no inert-mass plateau)."""
    w = _quiet_real_world()
    w.spawn_snake(0, mass=real_mass(256, 0.5))
    st = w.get_snake_states()[0]
    assert st.segment_count == 256  # at the physics cap
    w._food.spawn_at(st.head_x, st.head_y, 6.25)
    res = w.step({0: (1.0, 0.0, False)})
    assert res[0].mass_delta == pytest.approx(6.25 * w.get_config().pellet_mass_per_value)


def test_visible_state_emits_law_consistent_sct_and_fam() -> None:
    """R3 deliverable (4): VisibleState carries the client-law integers, so
    sim obs and the deployment bridge (which reads the real client's sct/fam)
    finally mean the same thing."""
    w = _quiet_real_world()
    target = real_mass(30, 0.25)
    w.spawn_snake(0, mass=target)
    w.spawn_snake(1, mass=real_mass(15, 0.0))
    # Let snake 1's chain grow to its law length before observing it.
    for _ in range(40):
        w.step({0: (1.0, 0.0, False), 1: (1.0, 0.0, False)})
    mass0 = w.get_snake_states()[0].mass
    vs = visible_state_from_world(w, 0)
    assert (vs.sct, vs.fam) == sct_fam_from_mass(mass0)
    assert real_mass(vs.sct, vs.fam) == pytest.approx(mass0)
    (enemy,) = vs.enemies
    assert enemy.sct == sct_fam_from_mass(w.get_snake_states()[1].mass)[0]
    # Legacy world: pre-R3 behavior — raw segment_count, fam hard 0.0.
    lw = World(WorldConfig(), seed=3)
    lw.spawn_snake(0)
    lvs = visible_state_from_world(lw, 0)
    assert lvs.sct == lw.get_snake_states()[0].segment_count
    assert lvs.fam == 0.0


def test_bot_sct_law_spawns_land_on_the_drawn_sct_under_real_growth() -> None:
    """The env's sct -> spawn-mass inverse must use the LUT law, or every
    sized bot would spawn ~79x too small."""
    from slither_gym.rl.env_gym import SlitherGymEnv
    c = realistic_world_config()
    env = SlitherGymEnv(world_config=c, num_bots=12, max_ticks=100, seed=13,
                        bot_sct_law="lognormal")
    env.reset()
    states = env._world.get_snake_states()
    scts = [states[i].segment_count for i in range(1, 13)]
    assert all(2 <= s <= 256 for s in scts)
    assert max(scts) > 12  # lognormal(median 22) puts most draws well past
    # spawn-size — impossible if the legacy inverse mapped sct to tiny mass
    for i in range(1, 13):
        assert states[i].segment_count == sct_fam_from_mass(states[i].mass)[0]


def test_real_mass_continuous_interpolates_within_the_segment() -> None:
    assert real_mass_continuous(22.0) == real_mass(22, 0.0)
    assert real_mass(22, 0.0) < real_mass_continuous(22.5) < real_mass(23, 0.0)
