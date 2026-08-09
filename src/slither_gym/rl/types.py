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
    # --- Normalizers bound to the world's physics constants. ---
    # Defaults reproduce the previously hard-coded arithmetic exactly.
    # REQUIRED COMPANION to the speed recalibration, not optional: self_state[5]
    # and enemy row[27] carry RAW per-tick speed, so raising base/boost speed
    # 3.0/6.0 -> 4.525/9.5 would multiply two encoder inputs by 1.51x/1.58x with
    # no other change. Set this to WorldConfig.base_speed and the feature reads
    # 1.0 at base speed and 2.099 while boosting, whatever the calibration.
    speed_norm: float = 1.0
    # Divisor for danger_segments[:, 2] (enemy body half-width). MUST equal
    # max_possible_segment_radius(world_config): under the "real_sc" width law
    # the true max is 22.076, so leaving 20.0 would emit values above 1.0.
    danger_radius_norm: float = 20.0


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
