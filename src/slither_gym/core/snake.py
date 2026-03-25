import math

import numpy as np
from numpy.typing import NDArray

from slither_gym.core.types import SnakeState, WorldConfig


def compute_segment_radius(mass: float, config: WorldConfig) -> float:
    t = min(mass / config.max_mass, 1.0)
    return config.min_segment_radius + (config.max_segment_radius - config.min_segment_radius) * math.sqrt(t)


def compute_turn_rate(mass: float, config: WorldConfig) -> float:
    t = min(mass / config.max_mass, 1.0)
    return config.max_turn_rate - (config.max_turn_rate - config.min_turn_rate) * math.sqrt(t)


def _expected_segments(mass: float, config: WorldConfig) -> int:
    """How many segments a snake should have at a given mass."""
    return max(config.initial_segments, config.initial_segments + int(mass - config.initial_mass))


class SnakeManager:
    """
    Manages all snake state and movement.
    Operates on pre-allocated arrays owned by the caller (World).
    """

    def __init__(self, config: WorldConfig) -> None:
        self._config = config
        self._max_seg = config.max_segments_per_snake
        self._states: list[SnakeState] = []
        self._seg_starts = np.zeros(config.max_snakes, dtype=np.int32)
        self._seg_ends = np.zeros(config.max_snakes, dtype=np.int32)

    def spawn(
        self,
        snake_id: int,
        x: float,
        y: float,
        angle: float,
        segments: NDArray[np.float32],
    ) -> SnakeState:
        config = self._config
        start = snake_id * self._max_seg
        seg_count = config.initial_segments

        dx = -math.cos(angle) * config.segment_spacing
        dy = -math.sin(angle) * config.segment_spacing
        for i in range(seg_count):
            segments[start + i, 0] = x + dx * i
            segments[start + i, 1] = y + dy * i

        self._seg_starts[snake_id] = start
        self._seg_ends[snake_id] = start + seg_count

        radius = compute_segment_radius(config.initial_mass, config)
        turn_rate = compute_turn_rate(config.initial_mass, config)

        state = SnakeState(
            snake_id=snake_id,
            alive=True,
            mass=config.initial_mass,
            speed=config.base_speed,
            angle=angle,
            boosting=False,
            head_x=x,
            head_y=y,
            segment_count=seg_count,
            segment_radius=radius,
            turn_rate=turn_rate,
        )

        while len(self._states) <= snake_id:
            self._states.append(SnakeState(
                snake_id=-1, alive=False, mass=0, speed=0, angle=0,
                boosting=False, head_x=0, head_y=0, segment_count=0,
                segment_radius=0, turn_rate=0,
            ))
        self._states[snake_id] = state
        return state

    def move(
        self,
        snake_id: int,
        target_cos: float,
        target_sin: float,
        boost: bool,
        segments: NDArray[np.float32],
    ) -> None:
        state = self._states[snake_id]
        if not state.alive:
            return
        config = self._config

        # Turn toward target angle
        target_angle = math.atan2(target_sin, target_cos)
        angle_diff = target_angle - state.angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        if abs(angle_diff) > state.turn_rate:
            angle_diff = math.copysign(state.turn_rate, angle_diff)
        new_angle = state.angle + angle_diff

        # Boost
        if boost and state.mass > config.initial_mass:
            speed = config.boost_speed
            state.mass -= config.boost_mass_cost_per_tick
            state.mass = max(state.mass, config.initial_mass)
            state.boosting = True
        else:
            speed = config.base_speed
            state.boosting = False

        # Advance head
        new_hx = state.head_x + math.cos(new_angle) * speed
        new_hy = state.head_y + math.sin(new_angle) * speed

        start = int(self._seg_starts[snake_id])
        end = int(self._seg_ends[snake_id])
        max_end = start + self._max_seg
        seg_count = end - start
        spacing = config.segment_spacing

        # Set new head position
        segments[start, 0] = new_hx
        segments[start, 1] = new_hy

        # Each segment follows the one ahead: if farther than spacing,
        # pull it to exactly spacing distance. If closer, leave it.
        # This is the standard "follow the leader" chain model.
        if seg_count > 1:
            seg = segments[start:end]
            for i in range(1, seg_count):
                dx = seg[i, 0] - seg[i - 1, 0]
                dy = seg[i, 1] - seg[i - 1, 1]
                d = math.sqrt(dx * dx + dy * dy)
                if d > spacing:
                    seg[i, 0] = seg[i - 1, 0] + dx / d * spacing
                    seg[i, 1] = seg[i - 1, 1] + dy / d * spacing

        # Growth: increase segment count. New segment duplicates tail.
        # On the next tick, the head moves forward and the chain pulls —
        # the new tail segment just stays put (it's within spacing of
        # the one ahead), effectively lengthening the snake forward.
        desired_segs = min(_expected_segments(state.mass, config), self._max_seg)
        if desired_segs > seg_count and end < max_end:
            segments[end, 0] = segments[end - 1, 0]
            segments[end, 1] = segments[end - 1, 1]
            end += 1

        # Shrinking (boosting): drop tail segments
        if desired_segs < end - start:
            end = start + desired_segs

        self._seg_ends[snake_id] = end
        state.segment_count = end - start
        state.angle = new_angle
        state.head_x = new_hx
        state.head_y = new_hy
        state.speed = speed
        state.segment_radius = compute_segment_radius(state.mass, config)
        state.turn_rate = compute_turn_rate(state.mass, config)

    def grow(self, snake_id: int, amount: float, segments: NDArray[np.float32]) -> None:
        """Add mass. Segment count adjusts naturally during move()."""
        state = self._states[snake_id]
        if not state.alive:
            return
        config = self._config
        state.mass = min(state.mass + amount, config.max_mass)
        state.segment_radius = compute_segment_radius(state.mass, config)
        state.turn_rate = compute_turn_rate(state.mass, config)

    def kill(self, snake_id: int, segments: NDArray[np.float32]) -> list[tuple[float, float, float]]:
        state = self._states[snake_id]
        if not state.alive:
            return []

        start = int(self._seg_starts[snake_id])
        end = int(self._seg_ends[snake_id])
        config = self._config

        mass_ratio = min(state.mass / config.max_mass, 1.0)
        pellet_value = config.corpse_food_base + (config.corpse_food_scale - config.corpse_food_base) * math.sqrt(mass_ratio)

        corpse: list[tuple[float, float, float]] = []
        for i in range(start, end):
            corpse.append((
                float(segments[i, 0]),
                float(segments[i, 1]),
                pellet_value,
            ))

        state.alive = False
        state.segment_count = 0
        self._seg_ends[snake_id] = self._seg_starts[snake_id]
        return corpse

    def get_head_position(self, snake_id: int, segments: NDArray[np.float32]) -> NDArray[np.float32]:
        start = int(self._seg_starts[snake_id])
        result: NDArray[np.float32] = segments[start].copy()
        return result

    def get_segment_slice(self, snake_id: int) -> tuple[int, int]:
        return int(self._seg_starts[snake_id]), int(self._seg_ends[snake_id])

    def alive_ids(self) -> list[int]:
        return [s.snake_id for s in self._states if s.alive]

    def get_state(self, snake_id: int) -> SnakeState:
        return self._states[snake_id]
