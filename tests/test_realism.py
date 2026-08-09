"""Acceptance tests for the sim-realism calibration (units A1-A5, B1-B3, C1, C4, C5).

Two jobs:
  1. Prove the legacy defaults are byte-identical, so E9-E30 and
     frozen_eval_v1..v7 remain reproducible.
  2. Score the realistic preset against the MEASURED real-slither.io values in
     docs/REAL_GAME_DATA.md, with the tolerances pre-registered in the spec.
"""
import math

import numpy as np
import pytest

from slither_gym.core.realism import (
    REAL_TURN_RATE_BINS,
    TICK_HZ,
    realistic_world_config,
    sample_world_config,
)
from slither_gym.core.snake import (
    SnakeManager,
    compute_segment_radius,
    compute_turn_rate,
    max_possible_segment_radius,
)
from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World


# --------------------------------------------------------------------------
# 1. Legacy identity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mass", [10.0, 50.0, 256.0, 5000.0, 40000.0])
def test_legacy_laws_match_the_original_closed_forms(mass: float) -> None:
    """The pre-calibration formulas, transcribed literally."""
    c = WorldConfig()
    t = min(mass / c.max_mass, 1.0)
    assert compute_segment_radius(mass, c) == pytest.approx(
        c.min_segment_radius + (c.max_segment_radius - c.min_segment_radius) * math.sqrt(t),
        abs=1e-12,
    )
    assert compute_turn_rate(mass, c) == pytest.approx(
        c.max_turn_rate - (c.max_turn_rate - c.min_turn_rate) * math.sqrt(t), abs=1e-12
    )


def test_legacy_defaults_are_untouched() -> None:
    c = WorldConfig()
    assert (c.base_speed, c.boost_speed) == (3.0, 6.0)
    assert (c.max_turn_rate, c.min_turn_rate) == (0.15, 0.02)
    assert (c.map_radius, c.segment_spacing) == (3000.0, 5.0)
    assert c.turn_rate_law == "legacy" and c.body_width_law == "legacy"
    # A4 off: the boost multiplier must be exactly neutral by default.
    assert c.boost_turn_multiplier == 1.0
    # C4 off: legacy absolute food counts.
    assert c.food_density_per_1e6 is None
    assert c.randomize_physics is False
    # max_possible_segment_radius must reproduce the old hard-coded 20.0 that
    # world.py used for both the hash cell size and the broad-phase radius.
    assert max_possible_segment_radius(c) == c.max_segment_radius == 20.0


def test_unknown_law_names_fail_loudly() -> None:
    import dataclasses
    c = WorldConfig()
    with pytest.raises(ValueError):
        compute_turn_rate(10.0, dataclasses.replace(c, turn_rate_law="nope"))
    with pytest.raises(ValueError):
        compute_segment_radius(10.0, dataclasses.replace(c, body_width_law="nope"))


# --------------------------------------------------------------------------
# 2. A1/A2/A3/A4/A5 against the measurements
# --------------------------------------------------------------------------

def test_a1_a2_speeds_match_measured_units_per_second() -> None:
    c = realistic_world_config()
    assert c.base_speed * TICK_HZ == pytest.approx(181.5)   # measured (8-game)
    assert c.boost_speed * TICK_HZ == pytest.approx(373.0)  # measured (8-game)
    # The ratio is the cross-check: measured 373/181.5 = 2.0551.
    assert c.boost_speed / c.base_speed == pytest.approx(2.055, abs=0.005)


@pytest.mark.parametrize("sct,expected_rad_per_s", REAL_TURN_RATE_BINS)
def test_a3_turn_curve_matches_measured_p95_bins(sct: float, expected_rad_per_s: float) -> None:
    c = realistic_world_config()
    mass = c.initial_mass + (sct - c.initial_segments)
    assert compute_turn_rate(mass, c) * TICK_HZ == pytest.approx(expected_rad_per_s, abs=0.08)


def test_a3_endpoint_span_matches_the_real_1_9x_range() -> None:
    c = realistic_world_config()
    span = compute_turn_rate(10.0, c) / compute_turn_rate(256.0, c)
    assert span == pytest.approx(2.13, abs=0.02)
    # Over the six measured bin representatives the realized span is the ~1.9x
    # the measurement reports (the endpoint span is wider because sct=256 is an
    # extrapolation past the last bin).
    rates = [compute_turn_rate(c.initial_mass + (s - 10), c) for s, _ in REAL_TURN_RATE_BINS]
    assert max(rates) / min(rates) == pytest.approx(1.94, abs=0.05)


def test_a5_min_turn_radius_is_the_emergent_quotient() -> None:
    """A5 has no symbol of its own — it is the acceptance test for A1+A2+A3+A4."""
    c = realistic_world_config()
    w = compute_turn_rate(c.initial_mass, c)
    base_r = c.base_speed / w
    boost_r = c.boost_speed / (w * c.boost_turn_multiplier)
    # Absolutes land ~7% under the clean measured base radius (46 u), because
    # w_max is anchored on the per-bin p95 turn-rate table rather than on the
    # single-frame minimum-radius extreme. Deliberate and documented (U9).
    assert base_r == pytest.approx(42.6, abs=0.5)
    assert boost_r == pytest.approx(93.7, abs=0.5)   # measured band 99-106 u
    # The RATIO is the well-determined quantity, and it is hit exactly.
    assert boost_r / base_r == pytest.approx(2.2, abs=0.02)


def test_a4_is_a_mild_penalty_because_most_of_it_is_already_emergent() -> None:
    """Pins the magnitude, which two separate briefs got wrong in both directions.

    The original brief guessed a 1/1.7 = 0.59 angular penalty. The first-pass
    data implied a 1.234 BONUS. Both are wrong: turn radius is the emergent
    quotient speed/omega, so a constant angular rate already inflates the
    boosting radius by the full speed ratio (2.0551x). The measurement wants
    2.2x, so only the small residual 2.0551/2.2 = 0.934 is a real penalty.
    """
    c = realistic_world_config()
    assert 0.85 < c.boost_turn_multiplier < 1.0          # mild penalty, not 0.59
    naive_ratio = c.boost_speed / c.base_speed           # what a 1.0 multiplier gives
    assert naive_ratio < 2.2                             # ...which UNDER-shoots
    assert naive_ratio / c.boost_turn_multiplier == pytest.approx(2.2, abs=0.02)


def test_a4_boost_changes_the_turn_clamp_in_move() -> None:
    c = realistic_world_config()
    segments = np.zeros((c.max_snakes * c.max_segments_per_snake, 2), dtype=np.float32)

    def heading_after(boost: bool) -> float:
        mgr = SnakeManager(c)
        st = mgr.spawn(0, 0.0, 0.0, 0.0, segments)
        st.mass = 100.0  # must exceed initial_mass for boost to engage
        # Demand a 180 deg turn so the clamp, not the target, decides the step.
        mgr.move(0, -1.0, 0.0, boost, segments)
        return abs(mgr.get_state(0).angle)

    assert heading_after(True) == pytest.approx(
        heading_after(False) * c.boost_turn_multiplier, rel=1e-9
    )


def test_a4_is_inert_under_legacy_defaults() -> None:
    c = WorldConfig()
    segments = np.zeros((c.max_snakes * c.max_segments_per_snake, 2), dtype=np.float32)

    def heading_after(boost: bool) -> float:
        mgr = SnakeManager(c)
        st = mgr.spawn(0, 0.0, 0.0, 0.0, segments)
        st.mass = 100.0
        mgr.move(0, -1.0, 0.0, boost, segments)
        return abs(mgr.get_state(0).angle)

    assert heading_after(True) == heading_after(False)


def test_state_turn_rate_stays_the_unboosted_base_rate() -> None:
    """SnakeState.turn_rate must keep its old meaning for existing consumers."""
    c = realistic_world_config()
    segments = np.zeros((c.max_snakes * c.max_segments_per_snake, 2), dtype=np.float32)
    mgr = SnakeManager(c)
    st = mgr.spawn(0, 0.0, 0.0, 0.0, segments)
    st.mass = 100.0
    mgr.move(0, 1.0, 0.0, True, segments)
    assert mgr.get_state(0).boosting
    assert mgr.get_state(0).turn_rate == pytest.approx(compute_turn_rate(st.mass, c))


# --------------------------------------------------------------------------
# 3. B1/B3 body width and collision
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sct", [2, 10, 22, 100, 178, 256])
def test_b1_width_follows_the_measured_sc_law_exactly(sct: int) -> None:
    c = realistic_world_config()
    mass = c.initial_mass + (sct - c.initial_segments)
    sc = 1.0 + (max(sct, c.initial_segments) - 2.0) / 106.0  # sim floors at sct=10
    assert compute_segment_radius(mass, c) == pytest.approx(c.body_radius_base * sc)


def test_b1_span_approaches_the_measured_3_45x() -> None:
    c = realistic_world_config()
    lo = compute_segment_radius(c.initial_mass, c)                      # sct 10
    hi = compute_segment_radius(c.initial_mass + 246, c)                # sct 256
    assert hi / lo == pytest.approx(3.16, abs=0.02)
    # ...versus the legacy law's reachable span, which is nearly flat. This is
    # the real defect B1 fixes: the sim was too FLAT, not too fat.
    lc = WorldConfig()
    legacy_span = compute_segment_radius(256.0, lc) / compute_segment_radius(10.0, lc)
    assert legacy_span == pytest.approx(1.33, abs=0.01)


def test_b3_broadphase_reaches_the_widest_possible_body() -> None:
    """The narrow phase tests head_r + other_r, so the broad phase must reach it."""
    c = realistic_world_config()
    rmax = max_possible_segment_radius(c)
    assert rmax == pytest.approx(22.076, abs=0.01)
    assert rmax > c.max_segment_radius  # the legacy 20.0 would under-query
    # SpatialHash asserts radius <= cell_size, and only scans a 3x3 neighborhood.
    # cell_size == 4*rmax and query_radius <= 2*rmax, so both hold for any R0.
    world = World(c, seed=0)
    assert world._spatial._cell_size == pytest.approx(4 * rmax)
    assert 2 * rmax <= world._spatial._cell_size


# --------------------------------------------------------------------------
# 4. Discretization invariants (the B2/B3 coupling)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config", [WorldConfig(), realistic_world_config()])
def test_world_construction_invariants_hold(config: WorldConfig) -> None:
    World(config, seed=0)  # the asserts live in World.__init__


def test_gapped_body_is_rejected_not_silently_mismodelled() -> None:
    """The real 18-27u spacing needs capsule collision first; refuse it loudly."""
    import dataclasses
    c = dataclasses.replace(realistic_world_config(), segment_spacing=27.0)
    with pytest.raises(AssertionError, match="tunnel"):
        World(c, seed=0)


def test_oversized_boost_step_is_rejected() -> None:
    import dataclasses
    c = dataclasses.replace(realistic_world_config(), boost_speed=50.0)
    with pytest.raises(AssertionError, match="jump clean through"):
        World(c, seed=0)


def test_realistic_boost_step_cannot_skip_food() -> None:
    """A2's 9.325 u/tick step exceeds the legacy collect radius; C5 fixes it."""
    c = realistic_world_config()
    collect = c.collect_radius_base + c.collect_radius_mass_mult * compute_segment_radius(
        c.initial_mass, c
    )
    assert c.boost_speed <= 2 * collect


# --------------------------------------------------------------------------
# 5. C1/C4 world scale and food density
# --------------------------------------------------------------------------

def test_c1_map_radius_default_is_unchanged_scale_is_a_knob() -> None:
    """Recorded project decision: matching the real map radius is NOT a goal."""
    assert realistic_world_config().map_radius == 3000.0
    real = realistic_world_config(real_world_scale=True)
    # MEASURED (border hit at r = 14976.5), not the old grd=32550 assumption —
    # grd is the world CENTRE coordinate. No randomization range: it is known.
    assert real.map_radius == 15000.0
    assert (real.map_radius_min, real.map_radius_max) == (None, None)


def test_c2_turn_map_ratio_is_reported_not_pinned() -> None:
    c = realistic_world_config(real_world_scale=True)
    base_r = c.base_speed / compute_turn_rate(c.initial_mass, c)
    assert base_r / c.map_radius == pytest.approx(2.79e-3, rel=0.05)  # measured


@pytest.mark.parametrize("radius", [1000.0, 3000.0, 8000.0])
def test_c4_food_density_is_scale_free(radius: float) -> None:
    import dataclasses
    c = dataclasses.replace(realistic_world_config(), map_radius=radius, max_food=1_000_000)
    world = World(c, seed=0)
    expected = 62.0 * math.pi * radius ** 2 / 1e6
    assert world._food.alive_count() == pytest.approx(expected, rel=0.01)


def test_a2b_boost_ramp_matches_the_measured_spin_up() -> None:
    """Onset ramp 469 u/s^2 -> 0.2931 u/tick^2; full 181.5 -> 373 u/s spin-up
    in ceil((9.325-4.5375)/0.2931) = 17 ticks (~0.41 s). Legacy default: None."""
    c = realistic_world_config()
    assert c.boost_ramp_up_per_tick == pytest.approx(469.0 / 1600.0)
    ticks_to_full = math.ceil((c.boost_speed - c.base_speed) / c.boost_ramp_up_per_tick)
    assert ticks_to_full == 17  # ~0.41 s at 40 Hz, matching the measured ~0.40 s
    assert WorldConfig().boost_ramp_up_per_tick is None


def test_c3_food_value_is_the_measured_iqr_uniform() -> None:
    """Real pellet value clusters at ~5 (p25 4.8, p75 6.2, thin tail to ~14).
    A uniform draw cannot express the cluster+tail shape, so the preset ships
    the IQR as the uniform range (mean 5.5); the legacy 1-3 stays untouched."""
    c = realistic_world_config()
    assert (c.food_value_min, c.food_value_max) == (4.8, 6.2)
    lc = WorldConfig()
    assert (lc.food_value_min, lc.food_value_max) == (1.0, 3.0)


def test_d4_boost_cost_point_is_mid_band() -> None:
    """D4 is a band (-0.1..-0.5 mass/s); the deterministic point value sits at
    the band centre-of-belief 0.25 mass/s = 0.00625 mass/tick, not the cheapest
    edge, and the DR range still spans the full band."""
    c = realistic_world_config()
    assert c.boost_mass_cost_per_tick == pytest.approx(0.00625)
    assert c.boost_mass_cost_per_tick_min == pytest.approx(0.1 / 40)
    assert c.boost_mass_cost_per_tick_max == pytest.approx(0.5 / 40)
    # Legacy default untouched (10-50x too high, but frozen for E-series).
    assert WorldConfig().boost_mass_cost_per_tick == 0.125


def test_c4_legacy_food_regime_is_untouched() -> None:
    c = WorldConfig()
    assert World(c, seed=0)._food.alive_count() == c.max_food // 2


# --------------------------------------------------------------------------
# 6. Domain randomization over the unmeasured constants
# --------------------------------------------------------------------------

def test_randomization_is_inert_and_draws_no_rng_by_default() -> None:
    c = realistic_world_config()
    rng = np.random.default_rng(0)
    assert sample_world_config(c, rng) is c
    # The stream must be untouched, or every existing run's trajectory shifts.
    assert rng.uniform() == np.random.default_rng(0).uniform()


def test_randomization_covers_the_documented_uncertainty() -> None:
    c = realistic_world_config(real_world_scale=True, randomize_physics=True)
    draws = [sample_world_config(c, np.random.default_rng(i)) for i in range(300)]

    # map radius is measured now, so it must NOT be randomized.
    assert {d.map_radius for d in draws} == {15000.0}
    # A4 spans the reported 99-106 u boost-radius band.
    assert min(d.boost_turn_multiplier for d in draws) >= 0.892
    assert max(d.boost_turn_multiplier for d in draws) <= 0.955
    # D4 boost mass cost is BOUNDED, not point-measured: a band, never a point.
    assert min(d.boost_mass_cost_per_tick for d in draws) >= 0.1 / 40
    assert max(d.boost_mass_cost_per_tick for d in draws) <= 0.5 / 40
    # R0 is not measured at all.
    assert min(d.body_radius_base for d in draws) >= 6.5 * 0.7
    assert max(d.body_radius_base for d in draws) <= 6.5 * 1.3
    # Every sampled world must still satisfy the discretization invariants.
    for d in draws:
        World(d, seed=0)


def test_sampling_never_compounds_across_episodes() -> None:
    from slither_gym.rl.env_gym import SlitherGymEnv
    c = realistic_world_config(randomize_physics=True)
    env = SlitherGymEnv(world_config=c, num_bots=0, max_ticks=50, seed=7)
    seen = []
    for _ in range(5):
        env.reset()
        seen.append(env._world_config.body_radius_base)
    assert env._base_world_config is c
    assert all(6.5 * 0.7 <= v <= 6.5 * 1.3 for v in seen)
