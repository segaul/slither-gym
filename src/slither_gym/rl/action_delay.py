"""S4: action-latency FIFO for the externally-commanded (RL) action path.

The real control loop is a ~30 Hz client plus network RTT; the legacy sim
applies an RL action to the very next 40 Hz physics tick with zero latency.
`ActionDelayQueue` models that latency as a per-snake FIFO of pending
tick-level actions: the action applied at physics tick t is the one commanded
at tick t - delay. At spawn the queue is seeded with the snake's initial
heading and no boost, so the first `delay` ticks behave exactly as an
uncommanded snake (World's own default for a missing action).

This lives at the ENV layer (env_gym / env_parallel), not in World, because:
  * World.step receives bot and RL actions indistinguishably and its contract
    is "no RL concepts leak in here" -- it cannot know which snakes are
    externally commanded.
  * Bots/scripted opponents model OTHER PLAYERS, whose latency is already
    implicit in their observed behavior; they must not be delayed.
  * Per-episode DR resolution (sample_world_config) already happens at env
    reset, which is exactly where a per-episode delay must be re-sampled and
    the queue re-seeded.

delay == 0 bypasses the queue entirely (apply() returns the commanded action
object unchanged), keeping every pre-S4 run byte-identical.
"""

from __future__ import annotations

import math
from collections import deque

Action = tuple[float, float, bool]  # (cos, sin, boost) -- World.step's format


class ActionDelayQueue:
    """FIFO of pending tick-level actions for one externally-commanded snake."""

    def __init__(self, delay_ticks: int) -> None:
        if delay_ticks < 0:
            raise ValueError(f"action delay must be >= 0, got {delay_ticks}")
        self._delay = int(delay_ticks)
        self._queue: deque[Action] = deque()

    @property
    def delay_ticks(self) -> int:
        return self._delay

    def seed(self, heading: float) -> None:
        """Reset to `delay` copies of straight-ahead/no-boost at the spawn
        heading. Call at episode reset (and on respawn, if the delayed snake
        ever respawns mid-episode)."""
        self._queue.clear()
        if self._delay > 0:
            hold: Action = (math.cos(heading), math.sin(heading), False)
            self._queue.extend([hold] * self._delay)

    def apply(self, commanded: Action) -> Action:
        """Enqueue this tick's commanded action; return the action to apply
        this tick (the one commanded `delay` ticks ago)."""
        if self._delay == 0:
            return commanded
        self._queue.append(commanded)
        return self._queue.popleft()
