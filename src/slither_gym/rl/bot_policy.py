import math

import numpy as np
from numpy.typing import NDArray

from typing import Any

from slither_gym.core.types import WorldConfig


class BotPolicy:
    """
    Rule-based policy for non-RL agents.
    Stateless — computes action from observation alone.
    """

    def __init__(self, config: WorldConfig, rng: np.random.Generator, hunter_prob: float = 0.7) -> None:
        self._config = config
        self._rng = rng
        # Reduced from 0.2: bots were fleeing at such long range they never
        # got killed, making it impossible for the agent to learn kill mechanics.
        self._danger_distance = 0.08
        # Some bots are "hunters" — chase the nearest enemy headlong with boost
        # instead of fleeing. They die a lot, but they're the population that
        # actually crashes into the RL agent's body, generating kill experience.
        self._hunter_prob = hunter_prob
        self._is_hunter = float(self._rng.random()) < hunter_prob

    def act(self, obs: dict[str, NDArray[np.float32]], **kwargs: Any) -> NDArray[np.float32]:
        self_state = obs["self_state"]
        enemies = obs["enemies"]
        food = obs["food"]

        current_cos = float(self_state[2])
        current_sin = float(self_state[3])

        dir_cos = current_cos
        dir_sin = current_sin
        boost = 0.0

        # Find the nearest active enemy (used by both flee and hunt logic)
        nearest_dist = float("inf")
        nearest_x = 0.0
        nearest_y = 0.0
        for i in range(enemies.shape[0]):
            if enemies[i, 31] > 0.5:  # is_active
                ex = float(enemies[i, 0])
                ey = float(enemies[i, 1])
                dist = math.sqrt(ex * ex + ey * ey)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_x = ex
                    nearest_y = ey

        if self._is_hunter and nearest_dist < 0.5:
            # Hunter mode: charge directly at nearest enemy head. Boost when close.
            mag = math.sqrt(nearest_x * nearest_x + nearest_y * nearest_y)
            if mag > 0:
                dir_cos = nearest_x / mag
                dir_sin = nearest_y / mag
                if nearest_dist < 0.15:
                    boost = 1.0
            return _normalize_action(dir_cos, dir_sin, boost, self._rng)

        # Avoidant fallback (the original logic)
        found_danger = nearest_dist < self._danger_distance

        if found_danger:
            flee_x = -nearest_x
            flee_y = -nearest_y
            mag = math.sqrt(flee_x * flee_x + flee_y * flee_y)
            if mag > 0:
                dir_cos = flee_x / mag
                dir_sin = flee_y / mag
        else:
            found_food = False
            for i in range(food.shape[0]):
                fx = float(food[i, 0])
                fy = float(food[i, 1])
                if fx == 0.0 and fy == 0.0 and float(food[i, 2]) == 0.0:
                    break
                mag = math.sqrt(fx * fx + fy * fy)
                if mag > 0:
                    dir_cos = fx / mag
                    dir_sin = fy / mag
                    found_food = True
                    break

            if not found_food:
                angle = math.atan2(current_sin, current_cos)
                angle += float(self._rng.normal(0, 0.3))
                dir_cos = math.cos(angle)
                dir_sin = math.sin(angle)

        return _normalize_action(dir_cos, dir_sin, boost, self._rng)


def _normalize_action(dir_cos: float, dir_sin: float, boost: float, rng: np.random.Generator) -> NDArray[np.float32]:
    dir_cos += float(rng.normal(0, 0.1))
    dir_sin += float(rng.normal(0, 0.1))
    mag = math.sqrt(dir_cos * dir_cos + dir_sin * dir_sin)
    if mag > 0:
        dir_cos /= mag
        dir_sin /= mag
    return np.array([dir_cos, dir_sin, boost], dtype=np.float32)
