import math

from slither_gym.core.types import SnakeState, StepResult, WorldConfig


def compute_reward(
    result: StepResult,
    snake_state: SnakeState,
    config: WorldConfig,
) -> float:
    """Pure function. Computes scalar reward from a single tick's outcome."""
    reward: float = 0.0

    reward += result.mass_delta * 1.0
    reward += result.remains_eaten * 1.0
    reward += result.kill_count * 5.0
    reward += config.survival_bonus

    # E15: gorge bonus — eating a big chunk of corpse AT ONCE (= consuming prey/a kill) pays extra,
    # on top of the linear remains reward. Only big single-tick gulps qualify; slow pellet foraging
    # doesn't. Not farmable without a real corpse nearby (can't fabricate one except by killing).
    if config.gorge_bonus_coef != 0.0 and result.remains_eaten >= config.gorge_threshold:
        reward += config.gorge_bonus_coef * result.remains_eaten

    if not result.alive:
        reward += config.death_penalty

    dist_from_center = math.sqrt(snake_state.head_x ** 2 + snake_state.head_y ** 2)
    edge_ratio = dist_from_center / config.map_radius
    if edge_ratio > 0.8:
        reward -= 0.1 * ((edge_ratio - 0.8) / 0.2)

    return reward
