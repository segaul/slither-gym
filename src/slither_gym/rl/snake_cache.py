from __future__ import annotations

import math

from slither_gym.rl.types import EnemySnakeInfo


class SnakeCache:
    """
    Maintains stable slot assignments for tracked enemy snakes.
    The same snake stays in the same slot across steps until it dies,
    leaves range, or is evicted by a closer snake.
    """

    def __init__(self, max_slots: int = 16) -> None:
        self._max_slots = max_slots
        self._slots: dict[int, int] = {}       # snake_id -> slot_index
        self._slot_to_snake: dict[int, int] = {}  # slot_index -> snake_id

    def update(
        self,
        visible_snakes: dict[int, EnemySnakeInfo],
        self_x: float,
        self_y: float,
        perception_radius: float,
    ) -> dict[int, int]:
        """
        Update cache with this frame's visible snakes.
        Returns snake_id -> slot_index mapping for constructing the observation.
        """
        evict_radius = 2.0 * perception_radius

        # 1. Evict dead or distant snakes
        to_evict: list[int] = []
        for snake_id, slot in self._slots.items():
            if snake_id not in visible_snakes:
                to_evict.append(snake_id)
            else:
                info = visible_snakes[snake_id]
                dist = math.hypot(info.head_x - self_x, info.head_y - self_y)
                if dist > evict_radius:
                    to_evict.append(snake_id)

        for snake_id in to_evict:
            slot = self._slots.pop(snake_id)
            self._slot_to_snake.pop(slot, None)

        # 2. Existing tracked snakes keep their slots (already in self._slots)

        # 3. Add new snakes to empty slots, closest first
        #    Only consider snakes within perception_radius for new assignments.
        #    (Existing snakes get the 2x buffer, but re-entry requires being close.)
        uncached = [
            sid for sid in visible_snakes
            if sid not in self._slots
            and math.hypot(visible_snakes[sid].head_x - self_x,
                           visible_snakes[sid].head_y - self_y) <= perception_radius
        ]
        if uncached:
            uncached.sort(key=lambda sid: math.hypot(
                visible_snakes[sid].head_x - self_x,
                visible_snakes[sid].head_y - self_y,
            ))
            free_slots = sorted(
                set(range(self._max_slots)) - set(self._slot_to_snake.keys())
            )
            for sid, slot in zip(uncached, free_slots):
                self._slots[sid] = slot
                self._slot_to_snake[slot] = sid

            # Remove assigned from uncached list
            uncached = [sid for sid in uncached if sid not in self._slots]

        # 4. Capacity overflow: evict furthest cached for closer uncached
        if uncached:
            for new_sid in uncached:
                new_dist = math.hypot(
                    visible_snakes[new_sid].head_x - self_x,
                    visible_snakes[new_sid].head_y - self_y,
                )
                # Find furthest currently cached snake
                furthest_sid = None
                furthest_dist = 0.0
                for cached_sid in self._slots:
                    if cached_sid not in visible_snakes:
                        continue
                    d = math.hypot(
                        visible_snakes[cached_sid].head_x - self_x,
                        visible_snakes[cached_sid].head_y - self_y,
                    )
                    if d > furthest_dist:
                        furthest_dist = d
                        furthest_sid = cached_sid

                if furthest_sid is not None and new_dist < furthest_dist:
                    # Evict furthest, assign its slot to new snake
                    slot = self._slots.pop(furthest_sid)
                    self._slot_to_snake.pop(slot)
                    self._slots[new_sid] = slot
                    self._slot_to_snake[slot] = new_sid
                else:
                    break  # No more swaps possible

        return dict(self._slots)

    def reset(self) -> None:
        """Clear all slots. Called on agent death/respawn and episode boundaries."""
        self._slots.clear()
        self._slot_to_snake.clear()
