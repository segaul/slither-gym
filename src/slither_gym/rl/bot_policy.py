import math

import numpy as np
from numpy.typing import NDArray

from slither_gym.core.types import WorldConfig


class BotPolicy:
    """
    Rule-based policy for non-RL agents.
    Stateless — computes action from observation alone.
    """

    def __init__(self, config: WorldConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._danger_distance = 0.2

    def act(self, obs: dict[str, NDArray[np.float32]]) -> NDArray[np.float32]:
        self_state = obs["self_state"]
        enemies = obs["enemies"]
        food = obs["food"]

        current_cos = float(self_state[2])
        current_sin = float(self_state[3])

        dir_cos = current_cos
        dir_sin = current_sin

        found_danger = False
        closest_head_dist = float("inf")
        closest_head_x = 0.0
        closest_head_y = 0.0

        for i in range(enemies.shape[0]):
            if enemies[i, 2] > 0.5:
                ex = float(enemies[i, 0])
                ey = float(enemies[i, 1])
                dist = math.sqrt(ex * ex + ey * ey)
                if dist < self._danger_distance and dist < closest_head_dist:
                    closest_head_dist = dist
                    closest_head_x = ex
                    closest_head_y = ey
                    found_danger = True

        if found_danger:
            flee_x = -closest_head_x
            flee_y = -closest_head_y
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

        dir_cos += float(self._rng.normal(0, 0.1))
        dir_sin += float(self._rng.normal(0, 0.1))

        mag = math.sqrt(dir_cos * dir_cos + dir_sin * dir_sin)
        if mag > 0:
            dir_cos /= mag
            dir_sin /= mag

        return np.array([dir_cos, dir_sin, 0.0], dtype=np.float32)
