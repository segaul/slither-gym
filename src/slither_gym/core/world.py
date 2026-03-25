import math

import numpy as np
from numpy.typing import NDArray

from slither_gym.core.food import FoodManager
from slither_gym.core.snake import SnakeManager, compute_segment_radius
from slither_gym.core.spatial_hash import SpatialHash
from slither_gym.core.types import SnakeState, StepResult, WorldConfig


class World:
    """
    Pure game state and physics. One tick = one call to step().
    No RL concepts leak in here.
    """

    def __init__(self, config: WorldConfig, seed: int = 0) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._tick: int = 0

        max_total = config.max_snakes * config.max_segments_per_snake
        self._segments = np.zeros((max_total, 2), dtype=np.float32)
        self._seg_alive = np.zeros(max_total, dtype=np.bool_)
        self._seg_owner = np.full(max_total, -1, dtype=np.int32)

        self._snakes = SnakeManager(config)
        self._food = FoodManager(config, self._rng)

        cell_size = compute_segment_radius(config.max_mass, config) * 4
        self._spatial = SpatialHash(
            cell_size=cell_size,
            bounds=config.map_radius,
        )

        self._food.spawn_batch(config.max_food // 2)

    def spawn_snake(self, snake_id: int, mass: float | None = None) -> None:
        """Spawn a snake at a random position within safe zone."""
        config = self._config
        angle = float(self._rng.uniform(0, 2 * math.pi))
        dist = float(config.map_radius * 0.8 * math.sqrt(self._rng.uniform(0, 1)))
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        heading = float(self._rng.uniform(0, 2 * math.pi))

        self._snakes.spawn(snake_id, x, y, heading, self._segments)

        if mass is not None and mass > config.initial_mass:
            # Add mass (segments grow naturally during move)
            self._snakes.grow(snake_id, mass - config.initial_mass, self._segments)
            # Pre-fill segment trail so the snake starts at full length
            # by running virtual moves straight ahead
            from slither_gym.core.snake import _expected_segments
            desired = min(_expected_segments(mass, config), config.max_segments_per_snake)
            state = self._snakes.get_state(snake_id)
            start = int(self._snakes._seg_starts[snake_id])
            # Place segments trailing behind head at spacing intervals
            dx = -math.cos(heading) * config.segment_spacing
            dy = -math.sin(heading) * config.segment_spacing
            actual_end = min(start + desired, start + config.max_segments_per_snake)
            for i in range(state.segment_count, actual_end - start):
                self._segments[start + i, 0] = x + dx * i
                self._segments[start + i, 1] = y + dy * i
            self._snakes._seg_ends[snake_id] = actual_end
            state.segment_count = actual_end - start

        start, end = self._snakes.get_segment_slice(snake_id)
        max_end = start + config.max_segments_per_snake
        self._seg_alive[start:end] = True
        self._seg_alive[end:max_end] = False
        self._seg_owner[start:end] = snake_id
        self._seg_owner[end:max_end] = -1

    def step(self, actions: dict[int, tuple[float, float, bool]]) -> dict[int, StepResult]:
        config = self._config
        alive_ids = self._snakes.alive_ids()
        if not alive_ids:
            self._tick += 1
            return {}

        initial_mass: dict[int, float] = {}
        for sid in alive_ids:
            initial_mass[sid] = self._snakes.get_state(sid).mass

        # 1. Move all snakes
        for sid in alive_ids:
            if sid in actions:
                cos_a, sin_a, boost = actions[sid]
            else:
                state = self._snakes.get_state(sid)
                cos_a = math.cos(state.angle)
                sin_a = math.sin(state.angle)
                boost = False
            self._snakes.move(sid, cos_a, sin_a, boost, self._segments)
            start, end = self._snakes.get_segment_slice(sid)
            max_end = start + config.max_segments_per_snake
            self._seg_alive[start:end] = True
            self._seg_alive[end:max_end] = False
            self._seg_owner[start:end] = sid
            self._seg_owner[end:max_end] = -1

        # 2. Rebuild spatial hash
        self._spatial.rebuild(self._segments, self._seg_alive, self._seg_owner)

        # 3. Check collisions
        killed: dict[int, int | None] = {}
        kill_counts: dict[int, int] = {sid: 0 for sid in alive_ids}

        for sid in alive_ids:
            state = self._snakes.get_state(sid)
            if not state.alive:
                continue

            dist_from_center = math.sqrt(state.head_x ** 2 + state.head_y ** 2)
            if dist_from_center > config.map_radius:
                killed[sid] = None
                continue

            head_radius = state.segment_radius
            query_radius = min(head_radius + config.max_segment_radius, self._spatial._cell_size)

            hits = self._spatial.query_near(
                state.head_x, state.head_y, query_radius, exclude_snake_id=sid
            )
            if hits:
                for seg_idx, other_sid in hits:
                    other_state = self._snakes.get_state(other_sid)
                    if not other_state.alive:
                        continue
                    dx = self._segments[seg_idx, 0] - state.head_x
                    dy = self._segments[seg_idx, 1] - state.head_y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < head_radius + other_state.segment_radius:
                        killed[sid] = other_sid
                        break

        # 4. Kill + spawn corpse food
        for sid, killer in killed.items():
            if killer is not None and killer not in killed:
                kill_counts[killer] = kill_counts.get(killer, 0) + 1

            corpse = self._snakes.kill(sid, self._segments)
            start = sid * config.max_segments_per_snake
            end = start + config.max_segments_per_snake
            self._seg_alive[start:end] = False
            self._seg_owner[start:end] = -1

            for fx, fy, fv in corpse:
                self._food.spawn_at(fx, fy, fv)

        # 5. Collect food
        remains_eaten: dict[int, float] = {sid: 0.0 for sid in alive_ids}
        for sid in self._snakes.alive_ids():
            state = self._snakes.get_state(sid)
            collect_radius = state.segment_radius * 3 + 10.0
            collected = self._food.collect_near(state.head_x, state.head_y, collect_radius)
            if collected > 0:
                self._snakes.grow(sid, collected, self._segments)
                start, end_new = self._snakes.get_segment_slice(sid)
                self._seg_alive[start:end_new] = True
                self._seg_owner[start:end_new] = sid
                remains_eaten[sid] = collected

        # 6. Batch-spawn food
        self._tick += 1
        if self._tick % config.food_refresh_interval == 0:
            self._food.spawn_batch(config.food_spawn_rate)

        # 7. Build results
        results: dict[int, StepResult] = {}
        for sid in alive_ids:
            state = self._snakes.get_state(sid)
            mass_delta = state.mass - initial_mass[sid]
            results[sid] = StepResult(
                alive=state.alive,
                mass_delta=mass_delta,
                killed_by=killed.get(sid),
                kill_count=kill_counts.get(sid, 0),
                remains_eaten=remains_eaten.get(sid, 0.0),
            )

        return results

    def get_snake_states(self) -> dict[int, SnakeState]:
        result: dict[int, SnakeState] = {}
        for state in self._snakes._states:
            if state.snake_id >= 0:
                result[state.snake_id] = state
        return result

    def get_segments(self, snake_id: int) -> NDArray[np.float32]:
        start, end = self._snakes.get_segment_slice(snake_id)
        result: NDArray[np.float32] = self._segments[start:end]
        return result

    def get_food_positions(self) -> NDArray[np.float32]:
        return self._food.get_alive_positions()

    def get_food_values(self) -> NDArray[np.float32]:
        return self._food.get_alive_values()

    def get_tick(self) -> int:
        return self._tick
