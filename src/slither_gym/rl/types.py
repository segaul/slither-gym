from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

AgentId = str  # "snake_0", "snake_1", etc.


@dataclass(frozen=True)
class EnemySnakeInfo:
    """Per-snake structured data for the enemies observation channel."""
    snake_id: int
    head_x: float
    head_y: float
    mass: float
    speed: float
    angle: float
    boosting: bool
    segments: NDArray[np.float32]  # (N, 2) all segments for this snake


@dataclass(frozen=True)
class ObsConfig:
    k_food: int = 64               # nearest floor food items
    k_enemies: int = 16            # tracked enemy snakes (was 128 segments)
    k_enemy_body_samples: int = 12 # body samples per tracked enemy
    k_own_body: int = 32           # sampled own body segments
    k_prey: int = 16               # nearest corpse food items
    k_danger_segments: int = 64    # nearest enemy body segments (collision radar)
    food_features: int = 3         # rel_x, rel_y, value
    prey_features: int = 3         # rel_x, rel_y, value
    enemy_features: int = 32       # head(2) + body_samples(24) + mass + speed + cos/sin(2) + boosting + is_active
    danger_features: int = 3       # rel_x, rel_y, radius
    own_body_features: int = 2     # rel_x, rel_y
    minimap_size: int = 64         # NxN grid covering the circular map


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
    self_segment_count: int      # NEW — number of body segments
    self_boosting: bool          # NEW — currently boosting?

    food_positions: NDArray[np.float32]  # (N, 2) absolute positions
    food_values: NDArray[np.float32]  # (N,)
    food_is_corpse: NDArray[np.bool_]  # (N,) NEW — True = corpse food

    own_segments: NDArray[np.float32]  # (K, 2) NEW — own body segment positions

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

    # Per-snake structured enemy data (for enemies channel)
    enemy_snakes: tuple[EnemySnakeInfo, ...] = ()
