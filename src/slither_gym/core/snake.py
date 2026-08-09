import math

import numpy as np
from numpy.typing import NDArray

from slither_gym.core.types import SnakeState, WorldConfig


def _expected_segments(mass: float, config: WorldConfig) -> int:
    """How many segments a snake should have at a given mass."""
    return max(config.initial_segments, config.initial_segments + int(mass - config.initial_mass))


def compute_segment_radius(mass: float, config: WorldConfig) -> float:
    """Body half-width. See WorldConfig.body_width_law."""
    if config.body_width_law == "legacy":
        t = min(mass / config.max_mass, 1.0)
        return config.min_segment_radius + (config.max_segment_radius - config.min_segment_radius) * math.sqrt(t)
    if config.body_width_law == "real_sc":
        # Keyed on the INTEGER segment count, so the collision disc can never
        # disagree with the number of segments actually rendered/indexed.
        sct = min(_expected_segments(mass, config), config.max_segments_per_snake)
        sc = 1.0 + (sct - config.body_radius_sct_offset) / config.body_radius_sct_divisor
        return config.body_radius_base * sc
    raise ValueError(f"unknown body_width_law: {config.body_width_law!r}")


def max_possible_segment_radius(config: WorldConfig) -> float:
    """Largest segment_radius any snake can reach under the active width law.

    Exists so collision broad-phase and observation normalizers can state their
    intent instead of relying on compute_segment_radius(max_mass, ...) happening
    to saturate — under "real_sc" that only works via the segment-count clamp.
    """
    if config.body_width_law == "legacy":
        return config.max_segment_radius
    sct = config.max_segments_per_snake
    sc = 1.0 + (sct - config.body_radius_sct_offset) / config.body_radius_sct_divisor
    return config.body_radius_base * sc


def compute_turn_rate(mass: float, config: WorldConfig) -> float:
    """Angular rate in rad/tick, un-boosted. See WorldConfig.turn_rate_law."""
    if config.turn_rate_law == "legacy":
        t = min(mass / config.max_mass, 1.0)
        return config.max_turn_rate - (config.max_turn_rate - config.min_turn_rate) * math.sqrt(t)
    if config.turn_rate_law == "real_sct":
        # Continuous in mass on purpose: _expected_segments' int() truncation
        # would put a 1-segment-wide staircase in the angular rate. Agrees with
        # _expected_segments exactly at integer mass.
        sct = config.initial_segments + (mass - config.initial_mass)
        span = config.turn_sct_ref_hi - config.turn_sct_ref_lo
        u = (sct - config.turn_sct_ref_lo) / span
        u = min(max(u, 0.0), 1.0)
        blend = float(u ** config.turn_rate_exponent)
        return config.max_turn_rate - (config.max_turn_rate - config.min_turn_rate) * blend
    raise ValueError(f"unknown turn_rate_law: {config.turn_rate_law!r}")


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

        # Hoisted above the turn clamp because the clamp depends on it (A4).
        #
        # P0.2 (state-space V5): boost ALWAYS engages when commanded. The old
        # predicate `boost and mass > initial_mass` refused boost at floor mass,
        # which dropped 99% of commanded boost ticks in real-capture replays
        # (humans boost constantly at sct 2-6, which maps to the sim's floor).
        # Real slither.io lets any snake at/above spawn mass boost; the floor
        # only stops the DRAIN, not the boost itself. That semantics lives in
        # the mass clamp below; above the floor, behaviour is identical.
        is_boosting = bool(boost)

        # Turn toward target angle
        target_angle = math.atan2(target_sin, target_cos)
        angle_diff = target_angle - state.angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        # state.turn_rate stays the UN-boosted base rate (tests and other
        # consumers depend on that meaning); the boost coupling lives only here.
        max_turn = state.turn_rate * (config.boost_turn_multiplier if is_boosting else 1.0)
        if abs(angle_diff) > max_turn:
            angle_diff = math.copysign(max_turn, angle_diff)
        new_angle = state.angle + angle_diff

        # Boost
        if is_boosting:
            target_speed = config.boost_speed
            # Drain clamps at spawn mass: boosting at the floor costs nothing
            # but still moves at boost_speed (P0.2, matches real slither.io).
            state.mass -= config.boost_mass_cost_per_tick
            state.mass = max(state.mass, config.initial_mass)
            state.boosting = True
        else:
            target_speed = config.base_speed
            state.boosting = False

        # A2b boost ramp: speed INCREASES are rate-limited (measured ~469 u/s^2
        # onset ramp); decreases are instant (measured: release drops to base
        # within one client frame). None = legacy instant onset, byte-identical.
        ramp = config.boost_ramp_up_per_tick
        if ramp is not None and target_speed > state.speed:
            speed = min(state.speed + ramp, target_speed)
        else:
            speed = target_speed

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

        corpse: list[tuple[float, float, float]] = []
        if config.corpse_value_law == "legacy":
            # Pre-P0.3 law, byte-identical: one pellet per segment worth
            # ~2.0-2.2, so a corpse returns roughly the victim's own mass.
            mass_ratio = min(state.mass / config.max_mass, 1.0)
            pellet_value = config.corpse_food_base + (config.corpse_food_scale - config.corpse_food_base) * math.sqrt(mass_ratio)
            for i in range(start, end):
                corpse.append((
                    float(segments[i, 0]),
                    float(segments[i, 1]),
                    pellet_value,
                ))
        elif config.corpse_value_law == "real":
            # D1/P0.3: total corpse value = corpse_mass_multiplier * mass
            # (measured: real corpses are worth ~20x what the legacy sim
            # drops). The per-segment budget is split into however many
            # pellets it takes for each pellet's value to land near
            # corpse_pellet_value_target (12.1, the midpoint of the observed
            # 10-14.2 high-value tail). value = budget / n conserves the
            # total EXACTLY regardless of rounding.
            seg_count = end - start
            if seg_count > 0:
                budget = config.corpse_mass_multiplier * state.mass / seg_count
                n = max(1, round(budget / config.corpse_pellet_value_target))
                pellet_value = budget / n
                for i in range(start, end):
                    x = float(segments[i, 0])
                    y = float(segments[i, 1])
                    for _ in range(n):
                        corpse.append((x, y, pellet_value))
        else:
            raise ValueError(f"unknown corpse_value_law: {config.corpse_value_law!r}")

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
