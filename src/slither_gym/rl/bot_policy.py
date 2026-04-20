"""
Rule-based bot policies that approximate real slither.io player behaviors.

Bot types:
  - FoodSeeker: Navigate toward food, avoid bodies. Basic cautious player.
  - Opportunist: Seek food normally, but boost to cut off smaller nearby snakes.
  - Circler: When near a smaller snake, turn perpendicular to try to encircle it.
  - Escaper: Avoid all enemies aggressively. Hard to kill, forces real trapping.
"""
import math

import numpy as np
from numpy.typing import NDArray

from typing import Any

from slither_gym.core.types import WorldConfig


class BotPolicy:
    """
    Rule-based policy for non-RL agents.
    Each bot is assigned a personality at init that determines its play style.
    """

    # Population mix: these should create diverse, realistic game dynamics
    PERSONALITIES = ["food_seeker", "opportunist", "circler", "escaper"]
    WEIGHTS = [0.40, 0.30, 0.20, 0.10]

    def __init__(self, config: WorldConfig, rng: np.random.Generator, **kwargs: Any) -> None:
        self._config = config
        self._rng = rng
        self._personality: str = rng.choice(self.PERSONALITIES, p=self.WEIGHTS)
        self._wander_angle: float = float(rng.uniform(0, 2 * math.pi))

    def act(self, obs: dict[str, NDArray[np.float32]], **kwargs: Any) -> NDArray[np.float32]:
        self_state = obs["self_state"]
        enemies = obs["enemies"]
        food = obs["food"]

        my_cos = float(self_state[2])
        my_sin = float(self_state[3])
        my_log_mass = float(self_state[4])  # log(mass / initial_mass)

        # Find nearest active enemy + its properties
        nearest = _find_nearest_enemy(enemies)

        if self._personality == "food_seeker":
            return self._act_food_seeker(my_cos, my_sin, nearest, food)
        elif self._personality == "opportunist":
            return self._act_opportunist(my_cos, my_sin, my_log_mass, nearest, food)
        elif self._personality == "circler":
            return self._act_circler(my_cos, my_sin, my_log_mass, nearest, food)
        elif self._personality == "escaper":
            return self._act_escaper(my_cos, my_sin, nearest, food)
        else:
            return self._act_food_seeker(my_cos, my_sin, nearest, food)

    def _act_food_seeker(
        self, my_cos: float, my_sin: float,
        nearest: dict, food: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Basic player: seek food, flee from nearby danger, never boost."""
        # Flee if enemy body is very close (danger)
        if nearest["dist"] < 0.12:
            return self._flee(nearest, boost=False)

        # Otherwise seek food
        direction = self._seek_food(my_cos, my_sin, food)
        if direction is not None:
            return _make_action(*direction, 0.0, self._rng)

        return self._wander(my_cos, my_sin)

    def _act_opportunist(
        self, my_cos: float, my_sin: float, my_log_mass: float,
        nearest: dict, food: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Seek food normally, but if a smaller snake is nearby, boost to cut it off.

        Cut-off strategy: move to a point AHEAD of the target (perpendicular
        intercept), not directly at its head. This creates body-wall kills
        instead of head-on suicides.
        """
        # Flee if very close
        if nearest["dist"] < 0.08:
            return self._flee(nearest, boost=True)

        # If enemy is nearby and smaller, try to cut it off
        if (nearest["dist"] < 0.3
                and nearest["active"]
                and nearest["log_mass"] < my_log_mass - 0.2):
            # Move to a point ahead of the enemy's heading
            # Enemy heading is (cos, sin) at indices 28-29
            target_x, target_y = _intercept_point(
                nearest["x"], nearest["y"],
                nearest["heading_cos"], nearest["heading_sin"],
                lead_distance=0.1,
            )
            mag = math.sqrt(target_x ** 2 + target_y ** 2)
            if mag > 0.01:
                boost = 1.0 if nearest["dist"] < 0.2 else 0.0
                return _make_action(target_x / mag, target_y / mag, boost, self._rng)

        # Otherwise seek food
        direction = self._seek_food(my_cos, my_sin, food)
        if direction is not None:
            return _make_action(*direction, 0.0, self._rng)

        return self._wander(my_cos, my_sin)

    def _act_circler(
        self, my_cos: float, my_sin: float, my_log_mass: float,
        nearest: dict, food: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """When near a smaller snake, turn perpendicular to try to encircle it.

        Circling strategy: move perpendicular to the vector toward the enemy,
        creating a curved body path around it. This is the real slither trap.
        """
        # Flee if very close and enemy is bigger
        if nearest["dist"] < 0.1 and nearest["log_mass"] > my_log_mass + 0.2:
            return self._flee(nearest, boost=True)

        # Circle if smaller enemy is within range
        if (nearest["dist"] < 0.25
                and nearest["active"]
                and nearest["log_mass"] < my_log_mass - 0.1):
            # Perpendicular direction (rotate enemy vector 90 degrees)
            # Choose rotation direction consistently (clockwise)
            perp_x = -nearest["y"]
            perp_y = nearest["x"]

            # Blend perpendicular with slight inward pull (spiral in)
            inward_x = nearest["x"]
            inward_y = nearest["y"]
            blend = 0.7  # mostly perpendicular, slightly inward
            dx = blend * perp_x + (1 - blend) * inward_x
            dy = blend * perp_y + (1 - blend) * inward_y
            mag = math.sqrt(dx * dx + dy * dy)
            if mag > 0.01:
                boost = 1.0 if nearest["dist"] < 0.15 else 0.0
                return _make_action(dx / mag, dy / mag, boost, self._rng)

        # Otherwise seek food
        direction = self._seek_food(my_cos, my_sin, food)
        if direction is not None:
            return _make_action(*direction, 0.0, self._rng)

        return self._wander(my_cos, my_sin)

    def _act_escaper(
        self, my_cos: float, my_sin: float,
        nearest: dict, food: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Prioritize avoiding all snakes. Boost away from danger. Hard to kill."""
        # Aggressive flee radius — escaper starts fleeing at longer range
        if nearest["dist"] < 0.25:
            return self._flee(nearest, boost=nearest["dist"] < 0.15)

        # Seek food cautiously
        direction = self._seek_food(my_cos, my_sin, food)
        if direction is not None:
            return _make_action(*direction, 0.0, self._rng)

        return self._wander(my_cos, my_sin)

    # --- Shared behaviors ---

    def _flee(self, nearest: dict, boost: bool = False) -> NDArray[np.float32]:
        flee_x = -nearest["x"]
        flee_y = -nearest["y"]
        mag = math.sqrt(flee_x * flee_x + flee_y * flee_y)
        if mag > 0:
            flee_x /= mag
            flee_y /= mag
        return _make_action(flee_x, flee_y, 1.0 if boost else 0.0, self._rng)

    def _seek_food(
        self, my_cos: float, my_sin: float,
        food: NDArray[np.float32],
    ) -> tuple[float, float] | None:
        """Find nearest food and return direction to it, or None."""
        for i in range(food.shape[0]):
            fx = float(food[i, 0])
            fy = float(food[i, 1])
            if fx == 0.0 and fy == 0.0 and float(food[i, 2]) == 0.0:
                break
            mag = math.sqrt(fx * fx + fy * fy)
            if mag > 0:
                return (fx / mag, fy / mag)
        return None

    def _wander(self, my_cos: float, my_sin: float) -> NDArray[np.float32]:
        self._wander_angle += float(self._rng.normal(0, 0.2))
        return _make_action(
            math.cos(self._wander_angle),
            math.sin(self._wander_angle),
            0.0,
            self._rng,
        )


def _find_nearest_enemy(enemies: NDArray[np.float32]) -> dict:
    """Extract nearest active enemy info from the enemies observation."""
    nearest: dict = {
        "dist": float("inf"), "x": 0.0, "y": 0.0,
        "log_mass": 0.0, "heading_cos": 1.0, "heading_sin": 0.0,
        "active": False,
    }
    for i in range(enemies.shape[0]):
        if enemies[i, 31] > 0.5:  # is_active
            ex = float(enemies[i, 0])
            ey = float(enemies[i, 1])
            dist = math.sqrt(ex * ex + ey * ey)
            if dist < nearest["dist"]:
                nearest["dist"] = dist
                nearest["x"] = ex
                nearest["y"] = ey
                nearest["log_mass"] = float(enemies[i, 26])
                nearest["heading_cos"] = float(enemies[i, 28])
                nearest["heading_sin"] = float(enemies[i, 29])
                nearest["active"] = True
    return nearest


def _intercept_point(
    target_x: float, target_y: float,
    heading_cos: float, heading_sin: float,
    lead_distance: float,
) -> tuple[float, float]:
    """Compute a point ahead of the target along its heading."""
    return (
        target_x + heading_cos * lead_distance,
        target_y + heading_sin * lead_distance,
    )


def _make_action(
    dir_cos: float, dir_sin: float, boost: float,
    rng: np.random.Generator,
    noise: float = 0.08,
) -> NDArray[np.float32]:
    dir_cos += float(rng.normal(0, noise))
    dir_sin += float(rng.normal(0, noise))
    mag = math.sqrt(dir_cos * dir_cos + dir_sin * dir_sin)
    if mag > 0:
        dir_cos /= mag
        dir_sin /= mag
    return np.array([dir_cos, dir_sin, boost], dtype=np.float32)
