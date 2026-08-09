from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Type aliases
Vec2 = NDArray[np.float32]  # shape (2,)
SegmentArray = NDArray[np.float32]  # shape (N, 2)


@dataclass(frozen=True)
class WorldConfig:
    map_radius: float = 3000.0
    max_snakes: int = 48
    max_segments_per_snake: int = 256
    max_food: int = 16384
    base_speed: float = 3.0  # u/tick; 120 u/s at 40Hz. Real: 181.5 u/s -> 4.5375 (realism.py)
    boost_speed: float = 6.0  # u/tick; 240 u/s. Real: 373 u/s -> 9.325 (realism.py)
    boost_mass_cost_per_tick: float = 0.125  # 5 segments/sec at 40Hz
    # Endpoints of the size->agility curve, in radians per tick. Their MEANING
    # depends on turn_rate_law (below):
    #   "legacy"   -> at minimum mass / at max_mass
    #   "real_sct" -> at sct == turn_sct_ref_lo / turn_sct_ref_hi
    # NOTE: scripts/analyze_probe.py and scripts/replay_dynamics.py hard-code
    # the legacy 0.15/0.02 and will print stale comparisons under "real_sct".
    max_turn_rate: float = 0.15
    min_turn_rate: float = 0.02
    segment_spacing: float = 5.0
    step_mul: int = 4  # physics ticks per RL decision
    initial_mass: float = 10.0
    initial_segments: int = 10
    max_mass: float = 40000.0
    min_segment_radius: float = 3.0
    max_segment_radius: float = 20.0
    food_value_min: float = 1.0
    food_value_max: float = 3.0
    food_spawn_rate: int = 50
    food_refresh_interval: int = 4
    corpse_food_base: float = 2.0  # min corpse pellet value
    corpse_food_scale: float = 8.0  # max corpse pellet value (for huge snakes)
    perception_radius: float = 500.0
    # --- E8 tunables (defaults = pre-E8 v1 behavior, byte-identical) ---
    collect_radius_base: float = 10.0  # additive collect radius
    collect_radius_mass_mult: float = 3.0  # collect radius per unit segment_radius
    survival_bonus: float = 0.01  # per-tick survival reward
    death_penalty: float = -10.0  # terminal reward on death (E9: rescale for scarce-food regime)
    # E11/E12: bot-difficulty curriculum for kill-discoverability. 1.0 = full realistic mix
    # (byte-identical to pre-E11, and what the eval always uses). Below 1.0, prob (1-difficulty)
    # of each bot being the `curriculum_prey` personality. Only matters when difficulty < 1.0.
    bot_difficulty: float = 1.0
    # Which exploitable personality the curriculum injects. E11 used "careless" (passive, REFUTED —
    # never collided). E12 default "kamikaze" (charges the nearest snake's body → forces the
    # collisions/kills the agent must sample). "careless" preserved for E11 reproducibility.
    curriculum_prey: str = "kamikaze"
    # E13: potential-based kill-credit shaping. coef=0.0 → OFF (byte-identical; eval/E9–E12 unchanged).
    # Adds r_shape = coef·(γ·Φ(s′) − Φ(s)) once per RL step, where Φ ∈ [0,1] = cut-readiness (an enemy
    # head close to + heading into one of my body segments). γ MUST match the trainer's RL γ.
    kill_shaping_coef: float = 0.0
    kill_shaping_gamma: float = 0.997  # must equal V4Config.gamma
    kill_shaping_radius: float = 120.0  # world-units: cut "proximity" scale (head-to-body distance)
    # E15: "gorge" bonus — extra reward for eating a LOT of corpse mass AT ONCE (= eating prey/kills),
    # not slow pellet foraging. Fires only when a single tick's corpse intake exceeds gorge_threshold.
    # coef=0 → OFF (byte-identical). Pulls the agent toward where snakes die (combat), making peaceful
    # foraging suboptimal — the E14 reform.
    gorge_bonus_coef: float = 0.0
    gorge_threshold: float = 5.0  # corpse-mass eaten in one tick that counts as a "big gulp" of prey
    # E24: reward per confirmed kill. Default 5.0 = byte-identical to E9–E23 (the +5 kill term).
    # Raising it strengthens the incentive to hunt (test: does more kill-reward → higher kill-rate).
    kill_reward_coef: float = 5.0

    # ------------------------------------------------------------------
    # Sim-realism calibration (2026-08-08). Measured constants and their
    # derivations: docs/REAL_GAME_DATA.md, docs/SIM_REALISM_STATE.md.
    #
    # EVERY DEFAULT BELOW REPRODUCES PRE-CALIBRATION BEHAVIOUR EXACTLY.
    # The measured values live in core/realism.py; apply them by building a
    # config with realism.realistic_world_config() rather than by editing
    # these defaults. E9-E30 and frozen_eval_v1..v7 must stay reproducible.
    # ------------------------------------------------------------------

    # --- A3: mass -> agility law ---
    # "legacy": turn = max - (max-min)*sqrt(mass/max_mass). Effectively flat over
    #   the reachable size range (5.92 -> 5.58 rad/s), because the knee of
    #   sqrt(mass/40000) sits ~40-150x beyond the 256-segment cap.
    # "real_sct": turn = max - (max-min)*u^q, u = clip((sct-lo)/(hi-lo), 0, 1).
    #   Fit to the measured p95 turn-rate-vs-sct table (real span 1.9x).
    turn_rate_law: str = "legacy"
    # Fit exponent q. Least-squares over the six measured sct bins
    # (grid search q in [0.30, 3.00] step 0.01, w_max/w_min solved linearly at
    # each q). Only read when turn_rate_law == "real_sct".
    turn_rate_exponent: float = 0.79
    # sct anchors for the real_sct blend. Deliberately NOT reusing
    # initial_segments / max_segments_per_snake so the fitted curve cannot move
    # silently if those unrelated fields are retuned (config-provenance).
    turn_sct_ref_lo: float = 10.0   # sct at which turn rate == max_turn_rate
    turn_sct_ref_hi: float = 256.0  # sct at which turn rate == min_turn_rate

    # --- A4: boost maneuverability ---
    # Angular-rate factor while boosting. READ THIS BEFORE "FIXING" ITS SIGN:
    # the sim clamps ANGULAR rate, so turn radius is the emergent quotient
    # speed/omega, and boosting ALREADY inflates the radius by the full speed
    # ratio (2.0551x under the realistic preset) with NO extra term. The
    # measured inflation is 46u -> 99-106u = ~2.2x, only slightly more, so the
    # residual angular penalty is mild: 2.0551 / 2.2 = 0.934 (realism.py).
    # The original brief's guess of a 1/1.7 = 0.59 penalty would double-count
    # the emergent part; a first-pass reading of 1.234 (a BONUS) came from a
    # boost-detection bug that has since been retracted.
    # 1.0 = legacy (boost has no effect on angular rate whatsoever).
    boost_turn_multiplier: float = 1.0

    # --- B1: body width law ---
    # "legacy": radius = min_segment_radius + (max-min)*sqrt(mass/max_mass).
    #   Reachable span is only 3.269 -> 4.362 u (1.33x) because 20.0 u is
    #   attained only at mass 40000, which the 256-segment cap makes
    #   unreachable. The defect is that the sim is too FLAT, not too fat.
    # "real_sc": radius = body_radius_base * sc, sc = 1 + (sct - offset)/divisor.
    #   The sc law is measured EXACTLY (zero residual, 94 distinct sizes,
    #   sct 2-262, real width span 3.45x).
    body_width_law: str = "legacy"
    # R0 = half-width in world units at sc == 1 (sct == 2). *** NOT MEASURED ***
    # The capture set gives only the dimensionless width RATIO, never an
    # absolute half-width. 6.5 is an ANCHOR, pinned by preserving the
    # scale-free ratio (spawn half-width)/(min turn radius) = 0.1634 across the
    # A-block recalibration. It assumes A1+A3 also apply; see realism.py.
    # Randomize it (body_radius_base_min/max) for any run whose conclusion
    # depends on absolute body width.
    body_radius_base: float = 6.5
    body_radius_sct_offset: float = 2.0     # measured: sc = 1 + (sct - 2)/106
    body_radius_sct_divisor: float = 106.0  # measured: exact over sct 2-262

    # --- A2b: boost acceleration ramp (P0.2 follow-on, measured on the F1 ruler) ---
    # Real slither.io does NOT jump to boost speed instantly: measured on the
    # 8-game captures, speed ramps ~linearly 181.5 -> ~380 u/s over ~12 client
    # frames (~0.40 s, ~469 u/s^2), while RELEASE drops to base speed within one
    # frame. The instant-onset model left every boost onset ~40 u AHEAD of the
    # real snake (2s pos err p50 45 u on boost windows vs 6.8 u non-boost).
    # None = legacy instant onset (byte-identical for E9-E30 / frozen evals).
    # When set: per-tick cap on speed INCREASE (u/tick per tick); decreases stay
    # instant. 469 u/s^2 at 40 Hz = 469/1600 = 0.2931 u/tick^2 (realism.py).
    boost_ramp_up_per_tick: float | None = None

    # --- C4: food density as a scale-free quantity ---
    # Pellets per 1e6 u^2. None = legacy absolute counts (max_food//2 at reset,
    # then food_spawn_rate pellets every food_refresh_interval ticks). When set,
    # the pellet population is held at density * pi * map_radius^2 / 1e6, so
    # food does not silently vanish or explode when map_radius is dialled.
    # Measured: ~62 per 1e6 u^2.
    food_density_per_1e6: float | None = None

    # --- C3/P0.3: food value distribution law ---
    # "legacy": one uniform(food_value_min, food_value_max) draw per pellet --
    #   byte-identical RNG stream to the pre-P0.3 sim.
    # "real": mixture matching the MEASURED distribution (min 3.0, p25 4.8,
    #   p50 5.2, p75 6.2, tail to 14.2, mean 6.25):
    #     prob (1 - food_tail_weight): lognormal(mu, sigma) clipped to
    #       [3.0, food_tail_lo]  (the bulk cluster at ~5.2)
    #     prob food_tail_weight: uniform(food_tail_lo, food_tail_hi)
    #       (the high-value 10-14.2 tail)
    #   Fit (realism.py FOOD_BULK_* note): mu=1.637, sigma=0.149, w=0.153 gives
    #   quartiles 4.74/5.32/6.14 and mean 6.253. The numeric fields below are
    #   inert unless food_value_law == "real".
    food_value_law: str = "legacy"
    food_bulk_log_mu: float = 1.637
    food_bulk_log_sigma: float = 0.149
    food_tail_lo: float = 10.0
    food_tail_hi: float = 14.2
    food_tail_weight: float = 0.153

    # --- D1/P0.3: corpse value law ---
    # "legacy": one pellet per segment worth
    #   corpse_food_base + (corpse_food_scale - base) * sqrt(mass/max_mass)
    #   ~= 2.0-2.2, so a corpse returns roughly the victim's own mass
    #   (~20 pellets' worth). Byte-identical to pre-P0.3.
    # "real": total corpse value = corpse_mass_multiplier * victim mass
    #   (measured: a real corpse is worth ~400 pellets vs the sim's ~20, i.e.
    #   ~20x the victim's mass), split evenly across segments, with
    #   pellets-per-segment chosen so each pellet's value lands near
    #   corpse_pellet_value_target -- the midpoint of the observed 10-14.2
    #   high-value tail, which is what real corpse pellets are. Total value is
    #   conserved exactly. Inert unless corpse_value_law == "real".
    corpse_value_law: str = "legacy"
    corpse_mass_multiplier: float = 20.0
    corpse_pellet_value_target: float = 12.1

    # --- S4: action latency (V5 plan, E5) ---
    # The real control loop is a ~30 Hz client plus network RTT; the sim
    # applies an RL action to the very next physics tick with zero latency.
    # action_delay_ticks delays the EXTERNALLY-COMMANDED (RL) action path by
    # this many physics ticks: the action applied at tick t is the one
    # commanded at tick t - delay, through a per-snake FIFO seeded with the
    # spawn heading / no-boost. Enforced at the ENV layer (env_gym /
    # env_parallel), not in World -- World cannot tell RL snakes from bots,
    # and bots model other players whose latency is already implicit in
    # their behavior, so they are never delayed.
    # 0 = legacy zero-latency, byte-identical to every pre-S4 run.
    # Scale: 0-2 client frames at 30 Hz ~= 0-66 ms ~= 0-3 sim ticks at 40 Hz.
    action_delay_ticks: int = 0

    # --- Domain randomization over UNMEASURED / UNCONFIRMED physics ---
    # Master gate. False => sample_world_config() is a no-op and draws no RNG,
    # so eval and every existing run stay bit-identical. Presets ship honest
    # ranges with this OFF; training turns it on, the frozen eval never does.
    randomize_physics: bool = False
    # Each range is (min, max); None on either side, or max <= min, means "not
    # randomized, use the point-valued field above".
    # map_radius: grd=32550 is ASSUMED to be the radius (docs/REAL_GAME_DATA.md
    # sec.6). The only two defensible readings are radius=grd and radius=grd/2,
    # so the realistic range spans exactly those two hypotheses.
    map_radius_min: float | None = None
    map_radius_max: float | None = None
    # boost_turn_multiplier: rests on exactly two scalars (40u / 68u) whose
    # sample sizes and size-composition are unreported. If the boosting sample
    # skews small the true value could be as low as 1.0 (no A4 at all).
    boost_turn_multiplier_min: float | None = None
    boost_turn_multiplier_max: float | None = None
    # body_radius_base: see the R0 note above -- this one is not measured at all.
    body_radius_base_min: float | None = None
    body_radius_base_max: float | None = None
    # segment_spacing: the real trend (~27u at sct 10 -> ~18u at sct 100) is
    # UNVERIFIED and its direction is physically backwards for a chain, so it is
    # suspected to be a capture-stride artifact. NOT adopted. See the
    # tube-continuity assert in World.__init__ before widening this.
    segment_spacing_min: float | None = None
    segment_spacing_max: float | None = None
    # boost_mass_cost_per_tick: BOUNDED, NOT POINT-MEASURED. Three reduction
    # methods disagree (-0.27..-0.52 / ~-0.11 / nonsense), giving a consensus
    # band of -0.1..-0.5 mass/s against the sim's -5.0 (10-50x too high). Not
    # further resolvable from natural play, so it ships as a range, never a
    # point estimate.
    boost_mass_cost_per_tick_min: float | None = None
    boost_mass_cost_per_tick_max: float | None = None
    # corpse_mass_multiplier: INFERRED from growth attribution (79% corpse
    # share, +520 length in one gorge), never counted from logged kill events.
    # Uncertainty is real and large -- the preset bands it +/-50% (10..30).
    corpse_mass_multiplier_min: float | None = None
    corpse_mass_multiplier_max: float | None = None
    # food_tail_weight: solved from the mean constraint (mean 6.25), not
    # counted directly. Preset band 0.10..0.20 (mixture mean ~5.9..~6.6).
    food_tail_weight_min: float | None = None
    food_tail_weight_max: float | None = None
    # action_delay_ticks: real RTT is UNMEASURED until the first live bridge
    # session (V5 plan S4/S7); the band covers 0-2 client frames at 30 Hz.
    # INTEGER range, sampled INCLUSIVE on both ends -- (0, 3) draws from
    # {0, 1, 2, 3} sim ticks, per-episode, constant within an episode.
    action_delay_ticks_min: int | None = None
    action_delay_ticks_max: int | None = None


@dataclass
class SnakeState:
    """Mutable state for one snake. Not a game object — just a data container."""
    snake_id: int
    alive: bool
    mass: float
    speed: float
    angle: float  # radians, heading direction
    boosting: bool
    head_x: float
    head_y: float
    segment_count: int
    segment_radius: float  # current width, computed from mass
    turn_rate: float  # current turn rate, computed from mass


@dataclass(frozen=True)
class StepResult:
    """What World.step() returns for each snake."""
    alive: bool
    mass_delta: float
    killed_by: int | None  # snake_id of killer, if died
    kill_count: int  # how many snakes this snake killed this tick
    remains_eaten: float  # mass gained from corpse food specifically
