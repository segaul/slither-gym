import math

import numpy as np
from numpy.typing import NDArray

from slither_gym.core.minimap import compute_minimap
from slither_gym.rl.types import ObsConfig, RawGameState


def compute_observation(
    state: RawGameState,
    obs_config: ObsConfig,
    snake_slot_mapping: dict[int, int] | None = None,
) -> dict[str, NDArray[np.float32]]:
    """
    Pure function. Same input always produces same output.
    Returns {"self_state": (12,), "food": (K_f, 3), "prey": (K_p, 3),
             "enemies": (K_e, 32), "danger_segments": (K_d, 3),
             "own_body": (K_b, 2), "minimap": (N, N)}.
    """
    initial_mass = 10.0
    max_segments = 256  # WorldConfig default
    perception_radius = 500.0

    # 1. Self state (12) — 8 original + 4 navigation compass
    # Nearest food/prey direction as explicit scalars so the network
    # can trivially learn "go toward food" without parsing 64 items.
    nearest_food_dx, nearest_food_dy = 0.0, 0.0
    nearest_prey_dx, nearest_prey_dy = 0.0, 0.0

    # Compute nearest food direction
    if len(state.food_positions) > 0:
        food_mask = ~state.food_is_corpse
        if np.any(food_mask):
            food_rel = state.food_positions[food_mask] - np.array([state.self_x, state.self_y])
            food_dists = np.sqrt(np.sum(food_rel * food_rel, axis=1))
            within = food_dists < perception_radius
            if np.any(within):
                nearest_idx = food_dists[within].argmin()
                nearest_food_dx = food_rel[within][nearest_idx, 0] / perception_radius
                nearest_food_dy = food_rel[within][nearest_idx, 1] / perception_radius

    # Compute nearest prey direction
    if len(state.food_positions) > 0:
        prey_mask = state.food_is_corpse
        if np.any(prey_mask):
            prey_rel = state.food_positions[prey_mask] - np.array([state.self_x, state.self_y])
            prey_dists = np.sqrt(np.sum(prey_rel * prey_rel, axis=1))
            within = prey_dists < perception_radius
            if np.any(within):
                nearest_idx = prey_dists[within].argmin()
                nearest_prey_dx = prey_rel[within][nearest_idx, 0] / perception_radius
                nearest_prey_dy = prey_rel[within][nearest_idx, 1] / perception_radius

    self_state = np.array([
        state.self_x / state.map_radius,
        state.self_y / state.map_radius,
        math.cos(state.self_angle),
        math.sin(state.self_angle),
        math.log(state.self_mass / initial_mass),
        state.self_speed,
        state.self_segment_count / max_segments,
        1.0 if state.self_boosting else 0.0,
        nearest_food_dx,
        nearest_food_dy,
        nearest_prey_dx,
        nearest_prey_dy,
    ], dtype=np.float32)

    # 2. Food observation (floor food only — corpse food goes to prey)
    food_obs = _compute_food(state, obs_config.k_food, perception_radius, corpse=False)

    # 3. Prey observation (corpse food only)
    prey_obs = _compute_food(state, obs_config.k_prey, perception_radius, corpse=True)

    # 4. Enemies observation — per-snake tracked (16, 32)
    enemy_obs = _compute_enemies(
        state, obs_config, perception_radius, initial_mass, snake_slot_mapping,
    )

    # 5. Danger segments — collision avoidance radar (64, 3)
    danger_obs = _compute_danger_segments(state, obs_config, perception_radius)

    # 6. Own body observation
    own_body_obs = _compute_own_body(state, obs_config, perception_radius)

    # 7. Minimap
    minimap = compute_minimap(
        state.all_snake_positions,
        state.all_snake_masses,
        state.map_radius,
        obs_config.minimap_size,
    )

    return {
        "self_state": self_state,
        "food": food_obs,
        "prey": prey_obs,
        "enemies": enemy_obs,
        "danger_segments": danger_obs,
        "own_body": own_body_obs,
        "minimap": minimap,
    }


def _compute_food(
    state: RawGameState,
    k: int,
    perception_radius: float,
    corpse: bool,
) -> NDArray[np.float32]:
    obs = np.zeros((k, 3), dtype=np.float32)
    if len(state.food_positions) == 0:
        return obs

    mask = state.food_is_corpse if corpse else ~state.food_is_corpse
    if not np.any(mask):
        return obs

    positions = state.food_positions[mask]
    values = state.food_values[mask]
    rel = positions - np.array([state.self_x, state.self_y], dtype=np.float32)
    dists = np.sqrt(np.sum(rel * rel, axis=1))

    within = dists < perception_radius
    if not np.any(within):
        return obs

    rel_in = rel[within]
    dists_in = dists[within]
    vals_in = values[within]

    order = np.argsort(dists_in)
    n_take = min(len(order), k)
    order = order[:n_take]

    obs[:n_take, 0] = rel_in[order, 0] / perception_radius
    obs[:n_take, 1] = rel_in[order, 1] / perception_radius
    obs[:n_take, 2] = vals_in[order]
    return obs


def _compute_enemies(
    state: RawGameState,
    obs_config: ObsConfig,
    perception_radius: float,
    initial_mass: float,
    snake_slot_mapping: dict[int, int] | None,
) -> NDArray[np.float32]:
    k_enemies = obs_config.k_enemies
    k_body = obs_config.k_enemy_body_samples
    enemy_obs = np.zeros((k_enemies, obs_config.enemy_features), dtype=np.float32)

    if not state.enemy_snakes:
        return enemy_obs

    # Build mapping if not provided (naive distance-sorted for bots/deployment)
    if snake_slot_mapping is None:
        sorted_snakes = sorted(
            state.enemy_snakes,
            key=lambda s: math.hypot(s.head_x - state.self_x, s.head_y - state.self_y),
        )
        snake_slot_mapping = {}
        for i, s in enumerate(sorted_snakes[:k_enemies]):
            snake_slot_mapping[s.snake_id] = i

    for info in state.enemy_snakes:
        slot = snake_slot_mapping.get(info.snake_id)
        if slot is None or slot >= k_enemies:
            continue

        row = enemy_obs[slot]

        # [0-1] head position relative to self
        row[0] = (info.head_x - state.self_x) / perception_radius
        row[1] = (info.head_y - state.self_y) / perception_radius

        # [2-25] 12 body samples x (dx, dy)
        segs = info.segments
        n_segs = len(segs)
        if n_segs > 0:
            if n_segs <= k_body:
                sampled = segs
                n_sampled = n_segs
            else:
                indices = np.linspace(0, n_segs - 1, k_body).astype(np.intp)
                sampled = segs[indices]
                n_sampled = k_body

            rel = sampled - np.array([state.self_x, state.self_y], dtype=np.float32)
            for j in range(n_sampled):
                row[2 + j * 2] = rel[j, 0] / perception_radius
                row[2 + j * 2 + 1] = rel[j, 1] / perception_radius

        # [26] log(mass / 10)
        row[26] = math.log(max(info.mass, 1.0) / initial_mass)
        # [27] speed
        row[27] = info.speed
        # [28-29] heading direction
        row[28] = math.cos(info.angle)
        row[29] = math.sin(info.angle)
        # [30] boosting
        row[30] = 1.0 if info.boosting else 0.0
        # [31] is_active
        row[31] = 1.0

    return enemy_obs


def _compute_danger_segments(
    state: RawGameState,
    obs_config: ObsConfig,
    perception_radius: float,
) -> NDArray[np.float32]:
    k = obs_config.k_danger_segments
    danger_obs = np.zeros((k, obs_config.danger_features), dtype=np.float32)

    if len(state.enemy_segments) == 0:
        return danger_obs

    rel = state.enemy_segments - np.array([state.self_x, state.self_y], dtype=np.float32)
    dists = np.sqrt(np.sum(rel * rel, axis=1))

    within = dists < perception_radius
    if not np.any(within):
        return danger_obs

    rel_in = rel[within]
    dists_in = dists[within]
    radius_in = state.enemy_segment_radius[within]

    order = np.argsort(dists_in)
    n_take = min(len(order), k)
    order = order[:n_take]

    danger_obs[:n_take, 0] = rel_in[order, 0] / perception_radius
    danger_obs[:n_take, 1] = rel_in[order, 1] / perception_radius
    danger_obs[:n_take, 2] = radius_in[order] / 20.0

    return danger_obs


def _compute_own_body(
    state: RawGameState,
    obs_config: ObsConfig,
    perception_radius: float,
) -> NDArray[np.float32]:
    k = obs_config.k_own_body
    own_body_obs = np.zeros((k, obs_config.own_body_features), dtype=np.float32)

    if len(state.own_segments) <= 1:
        return own_body_obs

    # Skip head (index 0) — agent knows head position from self_state
    body_segs = state.own_segments[1:]
    n_segs = len(body_segs)
    if n_segs == 0:
        return own_body_obs

    if n_segs <= k:
        sampled = body_segs
        n_take = n_segs
    else:
        indices = np.linspace(0, n_segs - 1, k, dtype=np.int32)
        sampled = body_segs[indices]
        n_take = k

    rel = sampled - np.array([state.self_x, state.self_y], dtype=np.float32)
    own_body_obs[:n_take, 0] = rel[:n_take, 0] / perception_radius
    own_body_obs[:n_take, 1] = rel[:n_take, 1] / perception_radius

    return own_body_obs
