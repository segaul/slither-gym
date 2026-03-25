from __future__ import annotations

import functools
import math
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv

from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World
from slither_gym.rl.obs_processor import compute_observation
from slither_gym.rl.reward import compute_reward
from slither_gym.rl.types import AgentId, ObsConfig, RawGameState


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
    ) -> None:
        super().__init__()
        self._world_config = world_config
        self._obs_config = obs_config
        self._num_agents = num_agents
        self._max_ticks = max_ticks
        self._seed = seed
        self._render_mode = render_mode

        self._world: World | None = None
        self._tick_count: int = 0

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
        self._world = World(self._world_config, seed=self._seed)
        self._tick_count = 0

        self.agents = list(self.possible_agents)

        for i in range(self._num_agents):
            self._world.spawn_snake(i)

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

        accumulated_rewards: dict[AgentId, float] = {agent: 0.0 for agent in self.agents}
        terminations: dict[AgentId, bool] = {agent: False for agent in self.agents}
        current_agents = list(self.agents)

        for _ in range(self._world_config.step_mul):
            results = self._world.step(world_actions)
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
        obs_config = self._obs_config
        return gymnasium.spaces.Dict({
            "self_state": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32,
            ),
            "food": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_config.k_food, obs_config.food_features),
                dtype=np.float32,
            ),
            "enemies": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_config.k_enemies, obs_config.enemy_features),
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

        states = self._world.get_snake_states()
        food_pos = self._world.get_food_positions()
        food_vals = self._world.get_food_values()

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
            enemy_is_head_list: list[bool] = []
            enemy_mass_list: list[float] = []
            enemy_speed_list: list[float] = []
            enemy_angle_list: list[float] = []
            enemy_radius_list: list[float] = []

            for other_idx, other_state in states.items():
                if other_idx == snake_idx or not other_state.alive:
                    continue
                segs = self._world.get_segments(other_idx)
                if len(segs) == 0:
                    continue
                enemy_segs_list.append(segs)
                n_segs = len(segs)
                enemy_is_head_list.extend([True] + [False] * (n_segs - 1))
                enemy_mass_list.extend([other_state.mass] * n_segs)
                enemy_speed_list.extend([other_state.speed] * n_segs)
                enemy_angle_list.extend([other_state.angle] * n_segs)
                enemy_radius_list.extend([other_state.segment_radius] * n_segs)

            if enemy_segs_list:
                all_enemy_segs = np.concatenate(enemy_segs_list, axis=0)
                all_is_head = np.array(enemy_is_head_list, dtype=np.bool_)
                all_mass = np.array(enemy_mass_list, dtype=np.float32)
                all_speed = np.array(enemy_speed_list, dtype=np.float32)
                all_angle = np.array(enemy_angle_list, dtype=np.float32)
                all_radius = np.array(enemy_radius_list, dtype=np.float32)
            else:
                all_enemy_segs = np.zeros((0, 2), dtype=np.float32)
                all_is_head = np.zeros(0, dtype=np.bool_)
                all_mass = np.zeros(0, dtype=np.float32)
                all_speed = np.zeros(0, dtype=np.float32)
                all_angle = np.zeros(0, dtype=np.float32)
                all_radius = np.zeros(0, dtype=np.float32)

            raw = RawGameState(
                self_x=state.head_x,
                self_y=state.head_y,
                self_mass=state.mass,
                self_speed=state.speed,
                self_angle=state.angle,
                food_positions=food_pos,
                food_values=food_vals,
                enemy_segments=all_enemy_segs,
                enemy_is_head=all_is_head,
                enemy_owner_mass=all_mass,
                enemy_owner_speed=all_speed,
                enemy_owner_angle=all_angle,
                enemy_segment_radius=all_radius,
                all_snake_positions=all_positions,
                all_snake_masses=all_masses,
                map_radius=self._world_config.map_radius,
            )

            observations[agent_id] = compute_observation(raw, self._obs_config)

        return observations

    def _empty_obs(self) -> dict[str, NDArray[np.float32]]:
        obs_config = self._obs_config
        return {
            "self_state": np.zeros(6, dtype=np.float32),
            "food": np.zeros((obs_config.k_food, obs_config.food_features), dtype=np.float32),
            "enemies": np.zeros((obs_config.k_enemies, obs_config.enemy_features), dtype=np.float32),
            "minimap": np.zeros((obs_config.minimap_size, obs_config.minimap_size), dtype=np.float32),
        }
