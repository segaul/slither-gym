from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

AgentId = str  # "snake_0", "snake_1", etc.


@dataclass(frozen=True)
class ObsConfig:
    k_food: int = 32
    k_enemies: int = 64
    food_features: int = 3  # rel_x, rel_y, value
    enemy_features: int = 7  # rel_x, rel_y, is_head, owner_mass, owner_speed, rel_velocity_angle, segment_radius
    minimap_size: int = 32  # NxN grid covering the circular map


@dataclass(frozen=True)
class RawGameState:
    """
    Minimal, source-agnostic game state.
    Produced by World (during training) or by TamperMonkey bridge (during deployment).
    """
    self_x: float
    self_y: float
    self_mass: float
    self_speed: float
    self_angle: float

    food_positions: NDArray[np.float32]  # (N, 2) absolute positions
    food_values: NDArray[np.float32]  # (N,)

    enemy_segments: NDArray[np.float32]  # (M, 2) absolute positions
    enemy_is_head: NDArray[np.bool_]  # (M,)
    enemy_owner_mass: NDArray[np.float32]  # (M,)
    enemy_owner_speed: NDArray[np.float32]  # (M,)
    enemy_owner_angle: NDArray[np.float32]  # (M,)
    enemy_segment_radius: NDArray[np.float32]  # (M,)

    # All snake head positions + masses for minimap
    all_snake_positions: NDArray[np.float32]  # (S, 2) absolute positions
    all_snake_masses: NDArray[np.float32]  # (S,)

    map_radius: float
