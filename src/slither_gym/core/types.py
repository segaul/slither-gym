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
    base_speed: float = 3.0
    boost_speed: float = 6.0
    boost_mass_cost_per_tick: float = 0.125  # 5 segments/sec at 40Hz
    max_turn_rate: float = 0.15  # radians per tick at minimum mass
    min_turn_rate: float = 0.02  # radians per tick at max mass
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
