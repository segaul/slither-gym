"""S5: the V5 observation path for the training envs (opt-in ``obs_schema='v5'``).

One shared implementation for env_gym and env_parallel: the env's world is
masked to the measured client delivery model (`visible_state_from_world`) and
fed to the ONE canonical `build_obs()` that deployment also uses — the whole
point of State Space V5 (P1). The V4 obs path (`obs_processor`) is untouched.

Last-action semantics (documented design choice)
------------------------------------------------
`self_state[9:12]` (last-action heading cos/sin + boost) and `self_state[7]`
(own boost) are filled from the agent's previous COMMANDED action — the value
the policy output last step, normalized, BEFORE the S4 action-delay queue —
not the delayed action the physics actually applied. Rationale: at deployment
the inference loop knows exactly what it commanded (it sent it), while the
applied-after-RTT action is unobservable client-side; training must condition
on the same signal (STATE_SPACE_V5_PLAN: "own boost = our own last commanded
action"). At episode reset the last-commanded is seeded with the spawn
heading and no boost, matching `ActionDelayQueue.seed`.
"""

from __future__ import annotations

import dataclasses
import math

import gymnasium
import numpy as np
from numpy.typing import NDArray

from slither_gym.core.world import World
from slither_gym.obs.schema_v5 import ObsConfigV5, build_obs
from slither_gym.obs.visibility import visible_state_from_world

# (cos, sin, boost) — the policy's normalized commanded action.
LastAction = tuple[float, float, float]


def v5_initial_last_action(spawn_angle: float) -> LastAction:
    """Reset seed: straight-ahead at the spawn heading, no boost (matches
    ActionDelayQueue.seed)."""
    return (math.cos(spawn_angle), math.sin(spawn_angle), 0.0)


def v5_observation_space(cfg: ObsConfigV5) -> gymnasium.spaces.Dict:
    """The declared space matches build_obs()'s shapes/dtypes exactly."""

    def box(shape: tuple[int, ...]) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
        )

    return gymnasium.spaces.Dict({
        "self_state": box((cfg.self_state_size,)),
        "food": box((cfg.k_food, 3)),
        "enemies": box((cfg.k_enemies, cfg.enemy_features)),
        "danger_segments": box((cfg.k_danger, 3)),
        "own_body": box((cfg.k_own_body, 2)),
    })


def v5_empty_obs(cfg: ObsConfigV5) -> dict[str, NDArray[np.float32]]:
    return {
        "self_state": np.zeros(cfg.self_state_size, dtype=np.float32),
        "food": np.zeros((cfg.k_food, 3), dtype=np.float32),
        "enemies": np.zeros(
            (cfg.k_enemies, cfg.enemy_features), dtype=np.float32
        ),
        "danger_segments": np.zeros((cfg.k_danger, 3), dtype=np.float32),
        "own_body": np.zeros((cfg.k_own_body, 2), dtype=np.float32),
    }


def v5_observe(
    world: World,
    snake_id: int,
    cfg: ObsConfigV5,
    last_action: LastAction,
) -> dict[str, NDArray[np.float32]]:
    """World -> VisibleState -> build_obs, with the commanded-action fills."""
    vs = visible_state_from_world(world, snake_id, cfg)
    # Own boost = our own last COMMANDED action (see module docstring); the
    # SnakeState.boosting flag is the delay-queue-applied one, which the
    # deployment loop cannot observe.
    vs = dataclasses.replace(vs, boosting=last_action[2] > 0.5)
    obs = build_obs(vs, cfg)
    obs["self_state"][9] = np.float32(last_action[0])
    obs["self_state"][10] = np.float32(last_action[1])
    obs["self_state"][11] = np.float32(last_action[2])
    return obs
