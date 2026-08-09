"""Measured real-slither.io constants, and the calibrated WorldConfig preset.

Everything here traces to `docs/REAL_GAME_DATA.md`. Values are the CONSOLIDATED
8-GAME set (13,492 ego frames, 1,947 snapshots, sct to 114) recorded on the state
board `docs/SIM_REALISM_STATE.md`, which SUPERSEDES the first 3-game figures.
Two of those first-pass numbers were artifacts of our own capture, and both are
corrected here -- see BOOST_TURN_MULTIPLIER and REAL_MAP_RADIUS.

WHY THIS IS A PRESET AND NOT A SET OF EDITED DEFAULTS
-----------------------------------------------------
E9-E30 and `frozen_eval_v1..v7` were all measured against the *dataclass
defaults* of `WorldConfig` -- and `env_factory.make_env` does not plumb most of
the physics fields, so those evals silently inherit whatever the defaults say.
Editing a default in place would therefore retroactively change what seven
frozen evals mean, with no diff touching them. That is a goalpost move. So the
defaults stay frozen and the calibration ships as `realistic_world_config()`.

UNITS
-----
The sim ticks physics at 40 Hz with `step_mul` physics ticks per RL decision
(in-repo confirmation: `boost_mass_cost_per_tick = 0.125` is documented as
"5 segments/sec at 40Hz", and 0.125 * 40 == 5.0). Rates measured in units/second
or radians/second are divided by TICK_HZ to become per-tick constants. Lengths
(radii, spacing, map radius, collect radius) are NOT converted -- they are
already in world units.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from slither_gym.core.types import WorldConfig

TICK_HZ = 40.0

# --- Kinematics (measured) -------------------------------------------------
# 8-game value. NOTE: base speed drifts +6% by sct 100, so it is not perfectly
# mass-independent as the 3-game pass reported. The sim's speed IS mass-
# independent (snake.py assigns config.base_speed unconditionally); modelling
# the drift is a separate unit and is NOT done here.
REAL_BASE_SPEED_PER_S = 181.5
REAL_BOOST_SPEED_PER_S = 373.0  # 8-game value; boost/base ratio 2.0551
# Min turn radius. The base figure is reported two ways: 40 u as a single-frame
# extreme and 46 u from the clean (boost-corrected) re-measure. The RATIO is the
# well-determined quantity, not the absolutes -- see BOOST_TURN_MULTIPLIER.
REAL_MIN_TURN_RADIUS = 46.0
REAL_MIN_TURN_RADIUS_BOOST_LO = 99.0
REAL_MIN_TURN_RADIUS_BOOST_HI = 106.0

# p95 turn rate (rad/s) per sct bin, each bin represented by the GEOMETRIC mean
# of its edges. Geometric, not arithmetic, because the enemy size distribution
# is strongly right-skewed (p10 2, p50 22, p90 178), so mass within a bin
# concentrates near the low edge.
# *** The per-bin mean sct is an ASSUMPTION, not data *** -- the measurement
# reports p95 per bin and never the mean sct within it. Re-reducing the 684
# world snapshots would settle it without a new capture.
REAL_TURN_RATE_BINS = (
    (4.5, 4.24),
    (15.8, 4.14),
    (35.4, 3.96),
    (70.7, 3.47),
    (141.4, 2.85),
    (228.9, 2.22),
)

# --- Body geometry (measured) ----------------------------------------------
# sc = 1 + (sct - 2)/106 -- EXACT, zero residual across 94 distinct sizes,
# sct 2-262. Real width span 1.0 -> 3.45.
REAL_SC_OFFSET = 2.0
REAL_SC_DIVISOR = 106.0

# --- World and food (measured) ---------------------------------------------
# *** CORRECTION vs the first pass. *** grd = 32550 was ASSUMED to be the map
# radius; it is actually the CENTRE coordinate (the world is centred at
# (grd, grd)). The radius is MEASURED at 15000 -- the red border was hit at
# r = 14976.5. So this is no longer an ambiguous quantity and is NOT randomized.
# Consequence: the sim is 2.4x tighter than real, not 5.3x.
REAL_MAP_RADIUS = 15000.0
REAL_WORLD_CENTRE = 32550.0  # grd; the real->sim bridge must subtract this
REAL_FOOD_DENSITY_PER_1E6 = 62.0  # pellets per 1e6 u^2
REAL_COLLECT_RADIUS = 75.0        # distance to nearest food when an eat fires

# --- Fitted A3 curve -------------------------------------------------------
# w(sct) = w_max - (w_max - w_min) * u^q,  u = clip((sct-10)/(256-10), 0, 1).
# Least squares over REAL_TURN_RATE_BINS (grid search q in [0.30, 3.00] step
# 0.01, w_max/w_min solved linearly at each q) gives 4.261 / 1.998 / 0.79.
# Rounded, the predictions are [4.260, 4.143, 3.885, 3.512, 2.883, 2.199] rad/s
# against measured [4.24, 4.14, 3.96, 3.47, 2.85, 2.22]; max |residual| 0.076
# rad/s = 3.4% of the 2.26 rad/s span.
FIT_TURN_RATE_MAX_PER_S = 4.26  # at sct == turn_sct_ref_lo (10)
FIT_TURN_RATE_MIN_PER_S = 2.00  # at sct == turn_sct_ref_hi (256) -- EXTRAPOLATED
FIT_TURN_RATE_EXPONENT = 0.79
TURN_SCT_REF_LO = 10.0
TURN_SCT_REF_HI = 256.0

# --- A4: boost angular-rate factor -----------------------------------------
# *** CORRECTION vs the first pass, and it FLIPS THE SIGN. *** The 3-game data
# said boosting raised the min turn radius only 40u -> 68u (1.70x) against a
# ~2.1x speed rise, which implied the real ANGULAR rate went UP while boosting
# (a bonus, multiplier 1.234). That was an artifact: boost detection used
# `sp > 6`, but base `sp` rises with size (5.78 at sct 0-19 -> 6.17 at 100-119),
# so every large snake read as permanently boosting and contaminated every
# boost-conditioned statistic. The clean 8-game value is 46u -> 99-106u = ~2.2x.
#
# Turn radius is the emergent quotient speed/omega, so with a constant angular
# rate the sim already inflates the boosting radius by exactly the speed ratio
# (2.0551x). The measurement wants 2.2x, i.e. slightly MORE than that, so the
# angular rate must DROP a little while boosting:
#   multiplier = speed_ratio / radius_ratio = 2.0551 / 2.2 = 0.934
# Range spans the reported 99-106 u radius band:
#   2.0551/2.304 = 0.892  ..  2.0551/2.152 = 0.955
# So A4 is a mild PENALTY after all -- but nothing like the 1/1.7 = 0.59 the
# original brief guessed, because most of the penalty is already emergent.
BOOST_TURN_MULTIPLIER = 0.934
BOOST_TURN_MULTIPLIER_LO = 0.892
BOOST_TURN_MULTIPLIER_HI = 0.955

# --- D4: boost mass cost (OUT OF THIS SPEC'S SCOPE, applied for coherence) ---
# Not one of the A/B/C units, but the sim's -5.0 mass/s is 10-50x the real
# drain, and leaving it while calibrating boost SPEED and boost TURNING would
# make the boost cost/benefit trade-off meaningless -- A4 would be scored
# against one measured number and one arbitrary one.
#
# BOUNDED, NOT POINT-MEASURED. Three methods disagree: negative-transitions
# gives -0.27..-0.52, shed-pellet counting ~ -0.11, and eat-lump subtraction
# returns nonsense (+0.27). Consensus band: -0.1 to -0.5 mass/s. The board
# records this as NOT further resolvable from natural play -- food density is
# high enough (one eat every 0.33 s while boosting) that no uncontaminated
# window exists. So it ships as a RANGE and never as a point estimate; the
# deterministic value is the conservative (cheapest-boost) end of the band.
REAL_BOOST_MASS_COST_PER_S_LO = 0.1
REAL_BOOST_MASS_COST_PER_S_HI = 0.5

# --- B1: R0, the one dimensional body-width quantity -----------------------
# *** NOT MEASURED. *** The capture set gives the width RATIO (1.0 -> 3.45) and
# never an absolute half-width in world units. R0 is pinned instead by holding a
# DIMENSIONLESS invariant of the pre-calibration sim across the A-block, so B1
# changes the SHAPE of width-vs-size without also silently rescaling absolute
# collision geometry:
#   invariant = (spawn half-width) / (min turn radius)
#   legacy    = 3.2688 / (3.0/0.15 = 20.0) = 0.16344
#   post-A1/A3 min turn radius = 4.5375 / 0.1065 = 42.61 u
#   target spawn half-width    = 0.16344 * 42.61 = 6.964 u
#   sc(sct=10) = 1 + (10-2)/106 = 1.075472  ->  R0 = 6.48  -> 6.5
# NOTE THE COUPLING: 6.5 assumes A1+A3 also apply. Without them the anchor
# would be 0.16344 * 20.0 / 1.075472 = 3.04. Do not mix.
BODY_RADIUS_BASE = 6.5
BODY_RADIUS_BASE_JITTER = 0.3  # +/-30%, for any run whose conclusion needs R0


def realistic_world_config(
    *,
    real_world_scale: bool = False,
    randomize_physics: bool = False,
    **overrides: object,
) -> WorldConfig:
    """WorldConfig with the measured real-slither.io physics applied.

    Never used implicitly -- an env only gets this if a config asks for it, so
    every pre-existing run and frozen eval is untouched.

    real_world_scale
        False (default) keeps `map_radius` at the sim's 3000. This follows the
        recorded project decision that WORLD SCALE IS A KNOB, NOT A CONSTANT:
        matching the real map radius is not a goal, and everything that depends
        on scale (food) is expressed as a density so the knob stays free. True
        adopts the MEASURED real radius of 15000, which costs 5x in
        map-traversal time per episode.
    randomize_physics
        Turns on per-episode sampling of the UNMEASURED constants (R0, the boost
        turn multiplier, R0, and the boost mass cost). Leave False for eval so
        the frozen eval stays deterministic.
    """
    fields: dict[str, object] = dict(
        # --- A1: 181 u/s measured / 40 Hz ---
        base_speed=REAL_BASE_SPEED_PER_S / TICK_HZ,      # 4.5375 u/tick
        # --- A2: 373 u/s measured / 40 Hz. boost/base = 2.0551, matching the
        # measured 373/181.5 exactly, so neither constant hides work ---
        boost_speed=REAL_BOOST_SPEED_PER_S / TICK_HZ,    # 9.325 u/tick
        # --- A3: refit size->agility curve ---
        turn_rate_law="real_sct",
        max_turn_rate=FIT_TURN_RATE_MAX_PER_S / TICK_HZ,  # 0.1065 rad/tick
        min_turn_rate=FIT_TURN_RATE_MIN_PER_S / TICK_HZ,  # 0.0500 rad/tick
        turn_rate_exponent=FIT_TURN_RATE_EXPONENT,
        turn_sct_ref_lo=TURN_SCT_REF_LO,
        turn_sct_ref_hi=TURN_SCT_REF_HI,
        # --- A4: measured 46u -> 99-106u min-radius rise (2.2x) against a
        # 2.0551x speed rise implies a mild ANGULAR penalty, 0.934x ---
        boost_turn_multiplier=BOOST_TURN_MULTIPLIER,
        # --- B1: sc = 1 + (sct - 2)/106, measured exact ---
        body_width_law="real_sc",
        body_radius_base=BODY_RADIUS_BASE,
        body_radius_sct_offset=REAL_SC_OFFSET,
        body_radius_sct_divisor=REAL_SC_DIVISOR,
        # --- C4: ~62 pellets per 1e6 u^2 measured ---
        food_density_per_1e6=REAL_FOOD_DENSITY_PER_1E6,
        # R1: hold the FLOOR population at that density. Counting corpse pellets
        # toward it let corpses switch floor spawning off for a whole episode.
        density_counts_floor_only=True,
        # D1/D3: the real economy is 79% corpse / 21% pellet, so the reward stream
        # must be able to tell them apart. Legacy `remains_eaten` counted all food,
        # double-paying floor pellets and making the corpse premium unrepresentable.
        remains_counts_corpse_only=True,
        # --- D6: measured real size distribution (sct p10 2 / p50 22 / p90 178 /
        # max 262). Opponents only. This is what makes a corpse worth killing for:
        # the premium comes from monsters existing, not from a corpse multiplier. ---
        spawn_mass_law="real_sct",
        # --- C5: ~75 u measured (distance to nearest food when an eat fires).
        # mass_mult is 0.0 because the measurement resolves a single radius and
        # says nothing about size-dependence -- inventing a slope would be
        # inventing a measurement. This also repairs the food-tunneling hole
        # that A2's 9.5 u/tick boost step would otherwise open (the legacy
        # collect radius of 8.27 u is smaller than one boosting step). ---
        collect_radius_base=REAL_COLLECT_RADIUS,
        collect_radius_mass_mult=0.0,
        randomize_physics=randomize_physics,
        # Uncertainty ranges. Inert unless randomize_physics is True.
        boost_turn_multiplier_min=BOOST_TURN_MULTIPLIER_LO,
        boost_turn_multiplier_max=BOOST_TURN_MULTIPLIER_HI,
        body_radius_base_min=BODY_RADIUS_BASE * (1.0 - BODY_RADIUS_BASE_JITTER),
        body_radius_base_max=BODY_RADIUS_BASE * (1.0 + BODY_RADIUS_BASE_JITTER),
        # D4 (see above): shipped as a band because the measurement is a lower
        # bound. The point value is the conservative (net) end.
        boost_mass_cost_per_tick=REAL_BOOST_MASS_COST_PER_S_LO / TICK_HZ,
        boost_mass_cost_per_tick_min=REAL_BOOST_MASS_COST_PER_S_LO / TICK_HZ,
        boost_mass_cost_per_tick_max=REAL_BOOST_MASS_COST_PER_S_HI / TICK_HZ,
    )
    if real_world_scale:
        # Now a MEASURED constant (border hit at r = 14976.5), so it is a point
        # value with no randomization range -- unlike the first pass, which had
        # to straddle two readings of grd.
        fields["map_radius"] = REAL_MAP_RADIUS
    # R2: `max_food` must follow the map, or C1 (map radius) and C4 (food density)
    # silently contradict each other. World._target_food_count clamps to
    # max_food*3//4, so at radius 15000 the 43,825-pellet target was delivered as
    # 12,288 -> 17.4 per 1e6 u^2 against the measured 62 (0.28x), while both units
    # were reported as clean wins. Sized from the density target itself so the two
    # cannot diverge again.
    #
    # MEASURED COST (25 snakes, 300 ticks): radius 15000 needs max_food 65536 to
    # reach density 62, which runs 240 ticks/s vs 442 at the old cap -- a 1.84x
    # throughput hit for ~0.9 MB. Memory is irrelevant; the time goes to
    # FoodManager.collect_near, which scans the whole pool per snake per tick.
    # A spatial index would remove it, but that is an optimization, not a fix.
    fields.setdefault("max_food", _max_food_for(fields))
    fields.update(overrides)
    return WorldConfig(**fields)  # type: ignore[arg-type]


def _max_food_for(fields: dict[str, object]) -> int:
    """Smallest power-of-two pool that can actually hold the density target."""
    density = float(fields.get("food_density_per_1e6") or 0.0)
    radius = float(fields.get("map_radius") or 0.0)
    default = int(WorldConfig.max_food)
    if density <= 0.0 or radius <= 0.0:
        return default
    target = density * 3.141592653589793 * radius * radius / 1e6
    needed = target / 0.75  # spawn_batch keeps a 25% free reserve for corpses
    size = default
    while size < needed:
        size *= 2
    return size


# Fields that carry a documented (min, max) uncertainty range.
_RANDOMIZED_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("map_radius", "map_radius_min", "map_radius_max"),
    ("boost_turn_multiplier", "boost_turn_multiplier_min", "boost_turn_multiplier_max"),
    ("body_radius_base", "body_radius_base_min", "body_radius_base_max"),
    ("segment_spacing", "segment_spacing_min", "segment_spacing_max"),
    (
        "boost_mass_cost_per_tick",
        "boost_mass_cost_per_tick_min",
        "boost_mass_cost_per_tick_max",
    ),
)


def sample_world_config(config: WorldConfig, rng: np.random.Generator) -> WorldConfig:
    """Resolve every randomized physics range to a concrete scalar.

    Call once per episode reset, BEFORE constructing World -- WorldConfig is
    frozen, so a constant that varied mid-episode would teleport geometry.

    Returns `config` unchanged and draws NO random numbers when
    `randomize_physics` is False (the default), so existing runs keep bit-
    identical RNG streams. The resolved-config writer should record the SAMPLED
    scalars, not the ranges, or the run is not reproducible.
    """
    if not config.randomize_physics:
        return config

    overrides: dict[str, Any] = {}
    for field, lo_name, hi_name in _RANDOMIZED_FIELDS:
        lo = getattr(config, lo_name)
        hi = getattr(config, hi_name)
        if lo is None or hi is None or hi <= lo:
            continue
        overrides[field] = float(rng.uniform(lo, hi))
    if not overrides:
        return config
    return dataclasses.replace(config, **overrides)
