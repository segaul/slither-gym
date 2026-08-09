from __future__ import annotations

import functools
import math
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from slither_gym.core.realism import sample_world_config
from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World
from slither_gym.obs.schema_v5 import ObsConfigV5
from slither_gym.rl.action_delay import ActionDelayQueue
from slither_gym.rl.env_obs_v5 import (
    LastAction,
    v5_empty_obs,
    v5_initial_last_action,
    v5_observation_space,
    v5_observe,
)
from slither_gym.rl.obs_processor import compute_observation
from slither_gym.rl.reward import compute_reward
from slither_gym.rl.types import AgentId, EnemySnakeInfo, ObsConfig, RawGameState


class SlitherParallelEnv(ParallelEnv):  # type: ignore[misc]
    """
    Multi-agent PettingZoo environment.
    All agents step simultaneously each tick.
    """

    metadata = {"name": "slither_v0", "render_modes": ["rgb_array"]}

    def __init__(
        self,
        world_config: WorldConfig = WorldConfig(),
        obs_config: ObsConfig = ObsConfig(),
        num_agents: int = 2,
        max_ticks: int = 3000,
        seed: int = 0,
        render_mode: str | None = None,
        obs_schema: str = "v4",
        obs_config_v5: ObsConfigV5 | None = None,
    ) -> None:
        super().__init__()
        if obs_schema not in ("v4", "v5"):
            raise ValueError(f"obs_schema must be 'v4' or 'v5', got {obs_schema!r}")
        # See SlitherGymEnv: base = pristine, _world_config = per-episode resolved.
        self._base_world_config = world_config
        self._world_config = world_config
        self._obs_config = obs_config
        # S5: opt-in deployable obs (schema_v5). 'v4' keeps the legacy
        # obs_processor path byte-identical.
        self._obs_schema = obs_schema
        self._obs_config_v5 = obs_config_v5 or ObsConfigV5()
        # S5: per-agent last COMMANDED action (see env_obs_v5 docstring).
        self._last_commanded: dict[int, LastAction] = {}
        self._num_agents = num_agents
        self._max_ticks = max_ticks
        self._seed = seed
        self._render_mode = render_mode

        self._world: World | None = None
        self._tick_count: int = 0
        # S4: per-agent latency FIFOs. EVERY agent here is externally
        # commanded (this env has no scripted bots), so all are delayed.
        self._action_delays: dict[int, ActionDelayQueue] = {}

        self.possible_agents: list[AgentId] = [
            f"snake_{i}" for i in range(num_agents)
        ]
        self.agents: list[AgentId] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        dict[AgentId, dict[str, NDArray[np.float32]]],
        dict[AgentId, dict[str, Any]],
    ]:
        if seed is not None:
            self._seed = seed
        # No-op unless randomize_physics is set. This env has no persistent RNG
        # of its own, so sample from a stream seeded off the episode seed —
        # reproducible from (resolved config, seed) alone.
        self._world_config = sample_world_config(
            self._base_world_config, np.random.default_rng(self._seed)
        )
        self._world = World(self._world_config, seed=self._seed)
        self._tick_count = 0

        self.agents = list(self.possible_agents)

        for i in range(self._num_agents):
            self._world.spawn_snake(i)

        # S4: per-episode delay (constant within the episode, re-sampled by
        # sample_world_config above when DR is on); each queue seeds with its
        # snake's spawn heading / no boost.
        states = self._world.get_snake_states()
        self._action_delays = {}
        self._last_commanded = {}
        for i in range(self._num_agents):
            q = ActionDelayQueue(self._world_config.action_delay_ticks)
            q.seed(states[i].angle)
            self._action_delays[i] = q
            # S5: seed last-commanded with the spawn heading / no boost,
            # exactly like the delay queue.
            self._last_commanded[i] = v5_initial_last_action(states[i].angle)

        observations = self._get_observations()
        infos: dict[AgentId, dict[str, Any]] = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self,
        actions: dict[AgentId, NDArray[np.float32]],
    ) -> tuple[
        dict[AgentId, dict[str, NDArray[np.float32]]],
        dict[AgentId, float],
        dict[AgentId, bool],
        dict[AgentId, bool],
        dict[AgentId, dict[str, Any]],
    ]:
        assert self._world is not None

        world_actions: dict[int, tuple[float, float, bool]] = {}
        for agent_id, action in actions.items():
            snake_idx = int(agent_id.split("_")[1])
            cos_a = float(action[0])
            sin_a = float(action[1])
            mag = math.sqrt(cos_a * cos_a + sin_a * sin_a)
            if mag > 0:
                cos_a /= mag
                sin_a /= mag
            else:
                cos_a = 1.0
                sin_a = 0.0
            boost = bool(action[2] > 0.5)
            world_actions[snake_idx] = (cos_a, sin_a, boost)
            # S5: record the COMMANDED (pre-delay-queue) action; the v5 obs
            # feeds this back as self_state[7] and [9:12] next step.
            self._last_commanded[snake_idx] = (
                cos_a, sin_a, 1.0 if boost else 0.0
            )

        accumulated_rewards: dict[AgentId, float] = {agent: 0.0 for agent in self.agents}
        terminations: dict[AgentId, bool] = {agent: False for agent in self.agents}
        current_agents = list(self.agents)

        for _ in range(self._world_config.step_mul):
            # S4: per-tick FIFO per agent — apply the action commanded
            # `delay` ticks ago. No-op when action_delay_ticks == 0.
            applied = {
                sid: self._action_delays[sid].apply(act)
                for sid, act in world_actions.items()
            }
            results = self._world.step(applied)
            self._tick_count += 1

            for agent_id in current_agents:
                snake_idx = int(agent_id.split("_")[1])
                if snake_idx in results:
                    result = results[snake_idx]
                    snake_state = self._world.get_snake_states().get(snake_idx)
                    if snake_state is not None:
                        reward = compute_reward(result, snake_state, self._world_config)
                        accumulated_rewards[agent_id] += reward
                    if not result.alive:
                        terminations[agent_id] = True
                        world_actions.pop(snake_idx, None)

        truncated = self._tick_count >= self._max_ticks
        truncations: dict[AgentId, bool] = {
            agent: truncated and not terminations[agent] for agent in self.agents
        }

        self.agents = [
            agent for agent in self.agents if not terminations[agent]
        ]

        observations = self._get_observations()
        infos: dict[AgentId, dict[str, Any]] = {
            agent: {} for agent in current_agents
        }

        for agent in current_agents:
            if agent not in observations:
                observations[agent] = self._empty_obs()

        return observations, accumulated_rewards, terminations, truncations, infos

    @functools.lru_cache(maxsize=1)
    def observation_space(self, agent: AgentId) -> gymnasium.spaces.Dict:
        if self._obs_schema == "v5":
            return v5_observation_space(self._obs_config_v5)
        obs_config = self._obs_config
        return gymnasium.spaces.Dict({
            "self_state": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32,
            ),
            "food": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_config.k_food, obs_config.food_features),
                dtype=np.float32,
            ),
            "prey": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_config.k_prey, obs_config.prey_features),
                dtype=np.float32,
            ),
            "enemies": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_config.k_enemies, obs_config.enemy_features),
                dtype=np.float32,
            ),
            "danger_segments": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_config.k_danger_segments, obs_config.danger_features),
                dtype=np.float32,
            ),
            "own_body": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_config.k_own_body, obs_config.own_body_features),
                dtype=np.float32,
            ),
            "minimap": gymnasium.spaces.Box(
                low=0.0, high=np.inf,
                shape=(obs_config.minimap_size, obs_config.minimap_size),
                dtype=np.float32,
            ),
        })

    @functools.lru_cache(maxsize=1)
    def action_space(self, agent: AgentId) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _get_observations(self) -> dict[AgentId, dict[str, NDArray[np.float32]]]:
        assert self._world is not None
        observations: dict[AgentId, dict[str, NDArray[np.float32]]] = {}

        # S5: the deployable obs path — visibility mask + canonical build_obs.
        if self._obs_schema == "v5":
            states_v5 = self._world.get_snake_states()
            for agent_id in self.agents:
                snake_idx = int(agent_id.split("_")[1])
                st = states_v5.get(snake_idx)
                if st is None or not st.alive:
                    continue
                observations[agent_id] = v5_observe(
                    self._world, snake_idx, self._obs_config_v5,
                    self._last_commanded[snake_idx],
                )
            return observations

        states = self._world.get_snake_states()
        food_pos = self._world.get_food_positions()
        food_vals = self._world.get_food_values()
        food_corpse = self._world.get_food_is_corpse()

        # Build minimap data (shared across all agents)
        alive_states = [s for s in states.values() if s.alive]
        if alive_states:
            all_positions = np.array([[s.head_x, s.head_y] for s in alive_states], dtype=np.float32)
            all_masses = np.array([s.mass for s in alive_states], dtype=np.float32)
        else:
            all_positions = np.zeros((0, 2), dtype=np.float32)
            all_masses = np.zeros(0, dtype=np.float32)

        for agent_id in self.agents:
            snake_idx = int(agent_id.split("_")[1])
            if snake_idx not in states or not states[snake_idx].alive:
                continue

            state = states[snake_idx]

            enemy_segs_list: list[NDArray[np.float32]] = []
            enemy_radius_list: list[float] = []
            enemy_snakes_list: list[EnemySnakeInfo] = []

            for other_idx, other_state in states.items():
                if other_idx == snake_idx or not other_state.alive:
                    continue
                segs = self._world.get_segments(other_idx)
                if len(segs) == 0:
                    continue
                enemy_segs_list.append(segs)
                enemy_radius_list.extend([other_state.segment_radius] * len(segs))
                enemy_snakes_list.append(EnemySnakeInfo(
                    snake_id=other_idx,
                    head_x=other_state.head_x,
                    head_y=other_state.head_y,
                    mass=other_state.mass,
                    speed=other_state.speed,
                    angle=other_state.angle,
                    boosting=other_state.boosting,
                    segments=segs,
                ))

            if enemy_segs_list:
                all_enemy_segs = np.concatenate(enemy_segs_list, axis=0)
                all_radius = np.array(enemy_radius_list, dtype=np.float32)
            else:
                all_enemy_segs = np.zeros((0, 2), dtype=np.float32)
                all_radius = np.zeros(0, dtype=np.float32)

            own_segs = self._world.get_segments(snake_idx)
            n_flat = len(all_enemy_segs)

            raw = RawGameState(
                self_x=state.head_x,
                self_y=state.head_y,
                self_mass=state.mass,
                self_speed=state.speed,
                self_angle=state.angle,
                self_segment_count=state.segment_count,
                self_boosting=state.boosting,
                food_positions=food_pos,
                food_values=food_vals,
                food_is_corpse=food_corpse,
                own_segments=own_segs,
                enemy_segments=all_enemy_segs,
                enemy_is_head=np.zeros(n_flat, dtype=np.bool_),
                enemy_owner_mass=np.zeros(n_flat, dtype=np.float32),
                enemy_owner_speed=np.zeros(n_flat, dtype=np.float32),
                enemy_owner_angle=np.zeros(n_flat, dtype=np.float32),
                enemy_segment_radius=all_radius,
                all_snake_positions=all_positions,
                all_snake_masses=all_masses,
                map_radius=self._world_config.map_radius,
                enemy_snakes=tuple(enemy_snakes_list),
            )

            observations[agent_id] = compute_observation(raw, self._obs_config)

        return observations

    def _empty_obs(self) -> dict[str, NDArray[np.float32]]:
        if self._obs_schema == "v5":
            return v5_empty_obs(self._obs_config_v5)
        obs_config = self._obs_config
        return {
            "self_state": np.zeros(12, dtype=np.float32),
            "food": np.zeros((obs_config.k_food, obs_config.food_features), dtype=np.float32),
            "prey": np.zeros((obs_config.k_prey, obs_config.prey_features), dtype=np.float32),
            "enemies": np.zeros((obs_config.k_enemies, obs_config.enemy_features), dtype=np.float32),
            "danger_segments": np.zeros((obs_config.k_danger_segments, obs_config.danger_features), dtype=np.float32),
            "own_body": np.zeros((obs_config.k_own_body, obs_config.own_body_features), dtype=np.float32),
            "minimap": np.zeros((obs_config.minimap_size, obs_config.minimap_size), dtype=np.float32),
        }
