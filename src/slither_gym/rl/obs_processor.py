import math

import numpy as np
from numpy.typing import NDArray

from slither_gym.core.minimap import compute_minimap
from slither_gym.rl.types import ObsConfig, RawGameState


def compute_observation(
    state: RawGameState,
    obs_config: ObsConfig,
) -> dict[str, NDArray[np.float32]]:
    """
    Pure function. Same input always produces same output.
    Returns {"self_state": (6,), "food": (K_f, 3), "enemies": (K_e, 7), "minimap": (N, N)}.
    """
    initial_mass = 10.0  # WorldConfig default

    # 1. Self state
    self_state = np.array([
        state.self_x / state.map_radius,
        state.self_y / state.map_radius,
        math.cos(state.self_angle),
        math.sin(state.self_angle),
        math.log(state.self_mass / initial_mass),
        state.self_speed,
    ], dtype=np.float32)

    perception_radius = 500.0  # WorldConfig default

    # 2. Food observation
    k_food = obs_config.k_food
    food_obs = np.zeros((k_food, obs_config.food_features), dtype=np.float32)

    if len(state.food_positions) > 0:
        rel_food = state.food_positions - np.array([state.self_x, state.self_y], dtype=np.float32)
        food_dists = np.sqrt(np.sum(rel_food * rel_food, axis=1))

        within = food_dists < perception_radius
        if np.any(within):
            rel_food_in = rel_food[within]
            food_dists_in = food_dists[within]
            food_vals_in = state.food_values[within]

            order = np.argsort(food_dists_in)
            n_take = min(len(order), k_food)
            order = order[:n_take]

            food_obs[:n_take, 0] = rel_food_in[order, 0] / perception_radius
            food_obs[:n_take, 1] = rel_food_in[order, 1] / perception_radius
            food_obs[:n_take, 2] = food_vals_in[order]

    # 3. Enemy observation with priority filtering
    k_enemies = obs_config.k_enemies
    enemy_obs = np.zeros((k_enemies, obs_config.enemy_features), dtype=np.float32)

    if len(state.enemy_segments) > 0:
        rel_enemy = state.enemy_segments - np.array([state.self_x, state.self_y], dtype=np.float32)
        enemy_dists = np.sqrt(np.sum(rel_enemy * rel_enemy, axis=1))

        within = enemy_dists < perception_radius
        if np.any(within):
            rel_in = rel_enemy[within]
            dists_in = enemy_dists[within]
            is_head_in = state.enemy_is_head[within]
            mass_in = state.enemy_owner_mass[within]
            speed_in = state.enemy_owner_speed[within]
            angle_in = state.enemy_owner_angle[within]
            radius_in = state.enemy_segment_radius[within]

            head_mask = is_head_in
            body_mask = ~is_head_in

            head_indices = np.where(head_mask)[0]
            body_indices = np.where(body_mask)[0]

            head_order = head_indices[np.argsort(dists_in[head_indices])] if len(head_indices) > 0 else np.array([], dtype=np.intp)
            body_order = body_indices[np.argsort(dists_in[body_indices])] if len(body_indices) > 0 else np.array([], dtype=np.intp)

            priority_order = np.concatenate([head_order, body_order])[:k_enemies]
            n_take = len(priority_order)

            sel = priority_order[:n_take]
            sel_rel = rel_in[sel]
            sel_angle = angle_in[sel]
            sel_mass = mass_in[sel]
            sel_speed = speed_in[sel]
            sel_is_head = is_head_in[sel]
            sel_radius = radius_in[sel]

            vec_to_self_angle = np.arctan2(-sel_rel[:, 1], -sel_rel[:, 0])
            rel_vel_diff = vec_to_self_angle - sel_angle
            rel_vel_angle = np.arctan2(np.sin(rel_vel_diff), np.cos(rel_vel_diff))

            enemy_obs[:n_take, 0] = sel_rel[:, 0] / perception_radius
            enemy_obs[:n_take, 1] = sel_rel[:, 1] / perception_radius
            enemy_obs[:n_take, 2] = sel_is_head.astype(np.float32)
            enemy_obs[:n_take, 3] = np.log(np.maximum(sel_mass, 1.0) / initial_mass)
            enemy_obs[:n_take, 4] = sel_speed
            enemy_obs[:n_take, 5] = rel_vel_angle
            enemy_obs[:n_take, 6] = sel_radius

    # 4. Minimap
    minimap = compute_minimap(
        state.all_snake_positions,
        state.all_snake_masses,
        state.map_radius,
        obs_config.minimap_size,
    )

    return {
        "self_state": self_state,
        "food": food_obs,
        "enemies": enemy_obs,
        "minimap": minimap,
    }
