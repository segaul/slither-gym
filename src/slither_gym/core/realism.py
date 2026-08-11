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
# A2b: boost ONSET is not instant. Median speed profile inside 94 long
# (>= 30-frame) boost runs ramps ~linearly 182 -> ~380 u/s over ~12 client
# frames (~0.40 s): slope (380 - 193.8) / (12 * 0.0331 s) = 469 u/s^2.
# RELEASE is instant (first post-boost frame is already back at ~182 u/s).
# A brief ~438 u/s overshoot at frames 16-20 is real in the median profile but
# NOT modelled -- one extra state variable for a ~15% transient. Measured
# 2026-08-09 on the 8-game captures via the F1 replay ruler.
REAL_BOOST_RAMP_PER_S2 = 469.0
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
# C3: pellet VALUE. Measured distribution is clustered at ~5.0 (p25 4.8,
# p75 6.2) with a thin tail out to ~14.6 (full range ~3-14). The config only
# supports a UNIFORM min..max draw, which cannot express "clustered with a
# tail": uniform(3, 14) would have mean 8.5, 70% too rich. So the preset ships
# the measured INTERQUARTILE band as the uniform range -- mean 5.5, matching
# where the real mass actually sits. Widening to the full 3-14 range needs a
# distribution law on FoodManager first (P0.3 follow-up), not a wider uniform.
REAL_FOOD_VALUE_P25 = 4.8
REAL_FOOD_VALUE_P75 = 6.2

# --- C3/P0.3: food value DISTRIBUTION LAW (supersedes the IQR-uniform stopgap)
# Measured pellet-value distribution (8-game consolidated set):
#   min 3.0, p25 4.8, p50 5.2, p75 6.2, tail to 14.2, MEAN 6.25.
# Uniform cannot express "cluster + tail" (uniform(3,14) has mean 8.5, ~70% too
# rich in equilibrium intake), so `food_value_law="real"` samples a MIXTURE:
#   with prob (1 - w): bulk ~ lognormal(mu, sigma), clipped to [3.0, tail_lo]
#   with prob w:       tail ~ uniform(tail_lo, tail_hi) = uniform(10.0, 14.2)
# FIT (fixed-point iteration; converged to 3 decimals):
#   1. Tail component: uniform(10, 14.2), mean 12.1 -- the observed high-value
#      cluster, which the capture notes attribute to corpse food mixed into the
#      floor population.
#   2. Given tail weight w, the mixture quartiles map to bulk quantiles
#      q/(1-w); least-squares fit of (mu, sigma) on ln{4.8, 5.2, 6.2} against
#      the corresponding normal z-scores.
#   3. w re-solved from the mean constraint
#      (1-w)*exp(mu + sigma^2/2) + w*12.1 = 6.25, repeat from 2.
# Converged: mu = 1.637, sigma = 0.149, w = 0.153.
#   Realized quartiles 4.74 / 5.32 / 6.14 (targets 4.8 / 5.2 / 6.2, max
#   residual 0.12); realized mean 6.253 (target 6.25); support [3.0, 14.2].
# w is INFERRED from the mean, not counted directly, so it carries a DR band.
FOOD_BULK_LOG_MU = 1.637
FOOD_BULK_LOG_SIGMA = 0.149
FOOD_TAIL_LO = 10.0
FOOD_TAIL_HI = 14.2
FOOD_TAIL_WEIGHT = 0.153
FOOD_TAIL_WEIGHT_LO = 0.10   # mixture mean ~5.90
FOOD_TAIL_WEIGHT_HI = 0.20   # mixture mean ~6.58

# --- D1/P0.3: corpse value ---------------------------------------------------
# Measured (3-game growth-attribution pass, docs/captains-log/2026-W32.md):
# 79% of real growth comes from corpse bursts; one gorge = +520 length in 8.6 s
# ~= 400 pellets' worth. The legacy sim corpse returns roughly the victim's own
# mass (~20 pellets' worth), i.e. real corpses are worth ~20x the sim's.
# In forage-time terms the D1 board row brackets the deficit at 5.6-20x.
# *** NOT POINT-MEASURED ***: the 400-pellet figure is inferred from growth
# attribution (enemies leaving view), not from logged kill events, and the
# real-mass LUT (fpsls/fmlts) was never applied to a corpse. So the multiplier
# ships at the top of the bracketed range with a GENEROUS +/-50% DR band.
# A future capture must log explicit death events with the victim's sct and the
# total mass the scavenger banked, to turn this band into a point.
CORPSE_MASS_MULTIPLIER = 20.0       # total corpse value = multiplier * victim mass
CORPSE_MASS_MULTIPLIER_LO = 10.0    # -50%
CORPSE_MASS_MULTIPLIER_HI = 30.0    # +50%
# Corpse pellets are the observed 10-14.2 tail of the value distribution;
# 12.1 is that band's midpoint, used to pick pellets-per-segment so each
# corpse pellet's value lands inside the measured tail.
CORPSE_PELLET_VALUE_TARGET = 12.1

# --- R3: growth law ----------------------------------------------------------
# The client converts eaten pellet value into segments through the superlinear
# fpsls/fmlts LUT law (core/growth.py; bitwise-verified vs the game01 capture,
# mscps=430). The legacy sim's 1-mass-per-segment conversion grew segments
# ~79x too fast at measured pellet values and hit the sct-256 physics cap
# ~23 s into a 200 s episode (docs/experiments/data/growth_law_r3.py) —
# the diagnosed E32 degenerate optimum. growth_law="real" tracks mass in the
# client's own currency; the two constants below are the only free numbers.
#
# Mass per unit of pellet value = measured growth per pellet (~1.3 mass/length
# units, docs/REAL_GAME_DATA.md sec. 4, 3-game growth-attribution pass)
# / measured mean pellet value (6.25, the C3 mixture mean) = 0.208.
# A RATIO OF TWO MEASUREMENTS (the 1.3 figure carries first-pass error), so it
# gets a +/-30% DR band; the LUT law itself is exact and is NOT randomized.
REAL_LENGTH_PER_PELLET = 1.3
PELLET_MASS_PER_VALUE = REAL_LENGTH_PER_PELLET / 6.25  # 0.208
PELLET_MASS_PER_VALUE_JITTER = 0.3
# D1 corpse multiplier RE-DERIVED for the real currency. The legacy-currency
# 20.0 ("~400 pellets vs the victim's ~20 mass") assumed pellet value == mass
# 1:1; in the client currency the anchor measurement is M3 (2026-W32): one
# gorge banked +520 mass from a sct-77 + sct-11 victim pair, whose real masses
# are real_mass(77,0) + real_mass(11,0) = 1419.2 + 149.4 = 1568.6. With
# banked = pellet_mass_per_value * multiplier * victim_mass, the multiplier is
#   520 / (0.208 * 1568.6) = 1.594
# (i.e. a fully-eaten corpse returns ~33% of the victim's real mass in the
# scavenger's currency — vs legacy-real's 20x * 0.208 = 4.16x, which would
# have made every kill worth 4 victims). Same honest +/-50% band as D1: the
# 520 figure is growth attribution, not a logged kill event.
CORPSE_MASS_MULTIPLIER_REAL = 1.594
CORPSE_MASS_MULTIPLIER_REAL_LO = 0.797   # -50%
CORPSE_MASS_MULTIPLIER_REAL_HI = 2.391   # +50%

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
# deterministic point value is the band's centre-of-belief 0.25 mass/s
# (0.00625 mass/tick — the V5 plan's mid-band figure), so a non-randomized
# run sits inside the band instead of pinned to its cheapest edge.
REAL_BOOST_MASS_COST_PER_S_LO = 0.1
REAL_BOOST_MASS_COST_PER_S_MID = 0.25
REAL_BOOST_MASS_COST_PER_S_HI = 0.5

# --- S4: action latency (V5 plan, E5) ---------------------------------------
# The real control loop is a ~30 Hz client (measured frame interval ~0.0331 s)
# plus network RTT; the sim's legacy behavior applies an action to the very
# next 40 Hz physics tick with zero latency. RTT is UNMEASURED until the first
# live bridge session (S7), so the band is the plan's stated envelope:
# 0-2 client frames = 0-66 ms = 0-3 sim ticks at 40 Hz, sampled per-episode.
# The deterministic point value is 1 tick (~1 client frame of pipeline delay,
# the minimum a real client ever has), so a non-randomized realistic run is
# not pinned to the physically-impossible zero-latency edge.
ACTION_DELAY_TICKS = 1
ACTION_DELAY_TICKS_LO = 0   # inclusive
ACTION_DELAY_TICKS_HI = 3   # inclusive

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
# R3 interaction: real-growth spawns are sct 2 (sc exactly 1.0), so the spawn
# half-width equals R0 itself. World.__init__'s anti-tunneling invariant
# (boost_speed <= 2 * min body radius) therefore clips the DR band's low edge
# at boost_speed/2 = 9.325/2 = 4.66 u; 4.68 leaves a hair of margin. Under the
# old sct-10 spawn (sc 1.0755) the -30% edge (4.55) never bound.
BODY_RADIUS_BASE_MIN_FLOOR = 4.68


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
        # --- A2b: measured 469 u/s^2 onset ramp -> 0.2931 u/tick^2 at 40 Hz.
        # Full 181.5 -> 373 spin-up takes (373-181.5)/469 = 0.41 s = 16 ticks ---
        boost_ramp_up_per_tick=REAL_BOOST_RAMP_PER_S2 / (TICK_HZ * TICK_HZ),
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
        # --- C3/P0.3: pellet value now a DISTRIBUTION LAW (cluster ~5.2 + tail
        # to 14.2, mean 6.25 -- see the FOOD_BULK_* fit note above). The
        # min/max fields stay set to the measured IQR purely as the fallback
        # for anything still reading them; the "real" law never does. ---
        food_value_law="real",
        food_value_min=REAL_FOOD_VALUE_P25,
        food_value_max=REAL_FOOD_VALUE_P75,
        food_bulk_log_mu=FOOD_BULK_LOG_MU,
        food_bulk_log_sigma=FOOD_BULK_LOG_SIGMA,
        food_tail_lo=FOOD_TAIL_LO,
        food_tail_hi=FOOD_TAIL_HI,
        food_tail_weight=FOOD_TAIL_WEIGHT,
        # --- R3: the client growth law. Mass is the client's own fpsls/fmlts
        # currency; pellet value converts at the measured 0.208 mass/value
        # (mean pellet ~1.3 mass, one segment >= 16.5 mass -> the real ~79x
        # slower segment growth and a ~minutes, not ~23 s, sct-256 cap) ---
        growth_law="real",
        pellet_mass_per_value=PELLET_MASS_PER_VALUE,
        # --- D1/P0.3: corpse value, in the REAL currency (see the R3 block:
        # multiplier 1.594 reproduces the measured +520-mass gorge from the
        # sct-77+11 pair; the legacy-currency 20.0 would overpay ~12x here).
        # Pellet VALUES still land inside the observed 10-14.2 tail ---
        corpse_value_law="real",
        corpse_mass_multiplier=CORPSE_MASS_MULTIPLIER_REAL,
        corpse_pellet_value_target=CORPSE_PELLET_VALUE_TARGET,
        # --- C5: ~75 u measured (distance to nearest food when an eat fires).
        # mass_mult is 0.0 because the measurement resolves a single radius and
        # says nothing about size-dependence -- inventing a slope would be
        # inventing a measurement. This also repairs the food-tunneling hole
        # that A2's 9.5 u/tick boost step would otherwise open (the legacy
        # collect radius of 8.27 u is smaller than one boosting step). ---
        collect_radius_base=REAL_COLLECT_RADIUS,
        collect_radius_mass_mult=0.0,
        # --- S4: 1 tick of action latency (~1 client frame), the minimum a
        # real 30 Hz client ever has; the DR band below covers RTT ---
        action_delay_ticks=ACTION_DELAY_TICKS,
        randomize_physics=randomize_physics,
        # Uncertainty ranges. Inert unless randomize_physics is True.
        boost_turn_multiplier_min=BOOST_TURN_MULTIPLIER_LO,
        boost_turn_multiplier_max=BOOST_TURN_MULTIPLIER_HI,
        body_radius_base_min=max(
            BODY_RADIUS_BASE * (1.0 - BODY_RADIUS_BASE_JITTER),
            BODY_RADIUS_BASE_MIN_FLOOR,  # R3: sct-2 spawns bind the
        ),                               # anti-tunneling invariant (see note)
        body_radius_base_max=BODY_RADIUS_BASE * (1.0 + BODY_RADIUS_BASE_JITTER),
        # P0.3/R3: corpse multiplier is inferred, not counted -- +/-50% band
        # around the real-currency point value (see the R3 block).
        corpse_mass_multiplier_min=CORPSE_MASS_MULTIPLIER_REAL_LO,
        corpse_mass_multiplier_max=CORPSE_MASS_MULTIPLIER_REAL_HI,
        # R3: mass-per-pellet-value is a ratio of two measurements -- +/-30%.
        pellet_mass_per_value_min=PELLET_MASS_PER_VALUE * (1.0 - PELLET_MASS_PER_VALUE_JITTER),
        pellet_mass_per_value_max=PELLET_MASS_PER_VALUE * (1.0 + PELLET_MASS_PER_VALUE_JITTER),
        # P0.3: tail weight is solved from the mean constraint, not counted.
        food_tail_weight_min=FOOD_TAIL_WEIGHT_LO,
        food_tail_weight_max=FOOD_TAIL_WEIGHT_HI,
        # D4 (see above): shipped as a band; the deterministic point value is
        # the mid-band 0.25 mass/s = 0.00625 mass/tick.
        boost_mass_cost_per_tick=REAL_BOOST_MASS_COST_PER_S_MID / TICK_HZ,
        boost_mass_cost_per_tick_min=REAL_BOOST_MASS_COST_PER_S_LO / TICK_HZ,
        boost_mass_cost_per_tick_max=REAL_BOOST_MASS_COST_PER_S_HI / TICK_HZ,
        # S4: RTT unmeasured until the first live bridge session -- the band
        # is the plan's 0-2 client-frame envelope (0-3 ticks, inclusive).
        action_delay_ticks_min=ACTION_DELAY_TICKS_LO,
        action_delay_ticks_max=ACTION_DELAY_TICKS_HI,
    )
    if real_world_scale:
        # Now a MEASURED constant (border hit at r = 14976.5), so it is a point
        # value with no randomization range -- unlike the first pass, which had
        # to straddle two readings of grd.
        fields["map_radius"] = REAL_MAP_RADIUS
    fields.update(overrides)
    return WorldConfig(**fields)  # type: ignore[arg-type]


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
    # P0.3: corpse value is inferred from growth attribution, not counted from
    # logged kill events -- the widest honest band in the file (+/-50%).
    ("corpse_mass_multiplier", "corpse_mass_multiplier_min", "corpse_mass_multiplier_max"),
    # P0.3: the food-value tail weight is solved from the mean constraint.
    ("food_tail_weight", "food_tail_weight_min", "food_tail_weight_max"),
    # R3: mass gained per unit pellet value is the ratio of two measurements
    # (1.3 length/pellet over mean value 6.25), not a client constant. The
    # fpsls/fmlts LUT law itself is exact and is deliberately NOT randomized.
    ("pellet_mass_per_value", "pellet_mass_per_value_min", "pellet_mass_per_value_max"),
)

# Integer-valued randomized fields: sampled with rng.integers, INCLUSIVE on
# both ends (a (0, 3) band draws uniformly from {0, 1, 2, 3}), unlike the
# float fields above which use rng.uniform over the open-ended interval.
_RANDOMIZED_INT_FIELDS: tuple[tuple[str, str, str], ...] = (
    # S4: action latency in sim ticks. RTT unmeasured; band = 0-2 client frames.
    ("action_delay_ticks", "action_delay_ticks_min", "action_delay_ticks_max"),
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
    for field, lo_name, hi_name in _RANDOMIZED_INT_FIELDS:
        lo = getattr(config, lo_name)
        hi = getattr(config, hi_name)
        if lo is None or hi is None or hi <= lo:
            continue
        # Inclusive on both ends: (0, 3) covers {0, 1, 2, 3}.
        overrides[field] = int(rng.integers(int(lo), int(hi) + 1))
    if not overrides:
        return config
    return dataclasses.replace(config, **overrides)
