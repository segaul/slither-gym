from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Type aliases
Vec2 = NDArray[np.float32]  # shape (2,)
SegmentArray = NDArray[np.float32]  # shape (N, 2)


@dataclass(frozen=True)
class WorldConfig:
    map_radius: float = 3000.0
    max_snakes: int = 32
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
