from __future__ import annotations

import math
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World
from slither_gym.rl.bot_policy import BotPolicy
from slither_gym.rl.env_parallel import SlitherParallelEnv
from slither_gym.rl.obs_processor import compute_observation
from slither_gym.rl.reward import compute_reward
from slither_gym.rl.types import AgentId, ObsConfig, RawGameState


class SlitherGymEnv(gymnasium.Env):  # type: ignore[type-arg]
    """
    Single-agent Gymnasium wrapper.
    The RL agent is always agent 0. All other agents are controlled by BotPolicy.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        world_config: WorldConfig = WorldConfig(),
        obs_config: ObsConfig = ObsConfig(),
        num_bots: int = 0,
        max_ticks: int = 3000,
        seed: int = 0,
        render_mode: str | None = None,
        respawn_bots: bool = True,
    ) -> None:
        super().__init__()
        self._world_config = world_config
        self._obs_config = obs_config
        self._num_bots = num_bots
        self._max_ticks = max_ticks
        self._seed = seed
        self._respawn_bots = respawn_bots
        self._render_mode = render_mode

        self._world: World | None = None
        self._rng = np.random.default_rng(seed)
        self._bot_policy = BotPolicy(world_config, self._rng)
        self._rl_agent_id: AgentId = "snake_0"
        self._tick_count: int = 0
        self._obs_update_counter: int = 0

        self._parallel_env = SlitherParallelEnv(
            world_config=world_config,
            obs_config=obs_config,
            num_agents=1 + num_bots,
            max_ticks=max_ticks,
            seed=seed,
        )

        self.observation_space = self._parallel_env.observation_space(self._rl_agent_id)
        self.action_space = self._parallel_env.action_space(self._rl_agent_id)

        self._bot_obs_cache: dict[int, dict[str, NDArray[np.float32]]] = {}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            self._bot_policy = BotPolicy(self._world_config, self._rng)

        self._world = World(self._world_config, seed=self._seed)
        self._tick_count = 0

        for i in range(1 + self._num_bots):
            self._world.spawn_snake(i)

        rl_obs = self._get_rl_observation()
        self._update_bot_obs_cache()
        return rl_obs, {}

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        assert self._world is not None
        config = self._world_config

        cos_a = float(action[0])
        sin_a = float(action[1])
        mag = math.sqrt(cos_a * cos_a + sin_a * sin_a)
        if mag > 0:
            cos_a /= mag
            sin_a /= mag
        else:
            cos_a, sin_a = 1.0, 0.0
        boost = bool(action[2] > 0.5)

        bot_actions: dict[int, tuple[float, float, bool]] = {}
        for i in range(1, 1 + self._num_bots):
            state = self._world.get_snake_states().get(i)
            if state is not None and state.alive:
                if i in self._bot_obs_cache:
                    bot_act = self._bot_policy.act(self._bot_obs_cache[i])
                    bot_actions[i] = (float(bot_act[0]), float(bot_act[1]), bool(bot_act[2] > 0.5))
                else:
                    bot_actions[i] = (math.cos(state.angle), math.sin(state.angle), False)

        total_reward = 0.0
        terminated = False

        for _ in range(config.step_mul):
            world_actions: dict[int, tuple[float, float, bool]] = {0: (cos_a, sin_a, boost)}
            world_actions.update(bot_actions)

            results = self._world.step(world_actions)
            self._tick_count += 1

            if 0 in results:
                result = results[0]
                snake_state = self._world.get_snake_states().get(0)
                if snake_state is not None:
                    reward = compute_reward(result, snake_state, config)
                    total_reward += reward
                if not result.alive:
                    terminated = True
                    break

        truncated = not terminated and self._tick_count >= self._max_ticks

        if self._respawn_bots:
            for i in range(1, 1 + self._num_bots):
                state = self._world.get_snake_states().get(i)
                if state is None or not state.alive:
                    self._world.spawn_snake(i)

        if terminated:
            rl_obs = self._empty_obs()
        else:
            rl_obs = self._get_rl_observation()

        if not terminated and not truncated:
            self._obs_update_counter += 1
            if self._obs_update_counter % 3 == 0:
                self._update_bot_obs_cache()

        info: dict[str, Any] = {}
        rl_state = self._world.get_snake_states().get(0)
        if rl_state is not None:
            info["mass"] = rl_state.mass

        return rl_obs, total_reward, terminated, truncated, info

    def _get_rl_observation(self) -> dict[str, NDArray[np.float32]]:
        assert self._world is not None
        states = self._world.get_snake_states()
        rl_state = states.get(0)
        if rl_state is None or not rl_state.alive:
            return self._empty_obs()

        food_pos = self._world.get_food_positions()
        food_vals = self._world.get_food_values()
        raw = self._build_raw_state(0, rl_state, states, food_pos, food_vals)
        return compute_observation(raw, self._obs_config)

    def _update_bot_obs_cache(self) -> None:
        assert self._world is not None
        self._bot_obs_cache.clear()
        states = self._world.get_snake_states()
        food_pos = self._world.get_food_positions()
        food_vals = self._world.get_food_values()

        for i in range(1, 1 + self._num_bots):
            bot_state = states.get(i)
            if bot_state is None or not bot_state.alive:
                continue
            raw = self._build_raw_state(i, bot_state, states, food_pos, food_vals)
            self._bot_obs_cache[i] = compute_observation(raw, self._obs_config)

    def _build_raw_state(
        self,
        snake_id: int,
        state: Any,
        all_states: dict[int, Any],
        food_pos: NDArray[np.float32] | None = None,
        food_vals: NDArray[np.float32] | None = None,
    ) -> RawGameState:
        assert self._world is not None
        if food_pos is None:
            food_pos = self._world.get_food_positions()
        if food_vals is None:
            food_vals = self._world.get_food_values()

        enemy_segs_list: list[NDArray[np.float32]] = []
        enemy_is_head_list: list[bool] = []
        enemy_mass_list: list[float] = []
        enemy_speed_list: list[float] = []
        enemy_angle_list: list[float] = []
        enemy_radius_list: list[float] = []

        # Also collect all snake positions for minimap
        all_positions_list: list[list[float]] = []
        all_masses_list: list[float] = []

        for other_id, other_state in all_states.items():
            if not other_state.alive:
                continue
            all_positions_list.append([other_state.head_x, other_state.head_y])
            all_masses_list.append(other_state.mass)

            if other_id == snake_id:
                continue
            segs = self._world.get_segments(other_id)
            if len(segs) == 0:
                continue

            dx = other_state.head_x - state.head_x
            dy = other_state.head_y - state.head_y
            if dx * dx + dy * dy > (self._world_config.perception_radius + 300) ** 2:
                continue

            enemy_segs_list.append(segs)
            n = len(segs)
            enemy_is_head_list.append(True)
            enemy_is_head_list.extend([False] * (n - 1))
            enemy_mass_list.extend([other_state.mass] * n)
            enemy_speed_list.extend([other_state.speed] * n)
            enemy_angle_list.extend([other_state.angle] * n)
            enemy_radius_list.extend([other_state.segment_radius] * n)

        if enemy_segs_list:
            all_enemy_segs: NDArray[np.float32] = np.concatenate(enemy_segs_list, axis=0)
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

        if all_positions_list:
            all_positions = np.array(all_positions_list, dtype=np.float32)
            all_masses = np.array(all_masses_list, dtype=np.float32)
        else:
            all_positions = np.zeros((0, 2), dtype=np.float32)
            all_masses = np.zeros(0, dtype=np.float32)

        return RawGameState(
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

    def _empty_obs(self) -> dict[str, NDArray[np.float32]]:
        obs_config = self._obs_config
        return {
            "self_state": np.zeros(6, dtype=np.float32),
            "food": np.zeros((obs_config.k_food, obs_config.food_features), dtype=np.float32),
            "enemies": np.zeros((obs_config.k_enemies, obs_config.enemy_features), dtype=np.float32),
            "minimap": np.zeros((obs_config.minimap_size, obs_config.minimap_size), dtype=np.float32),
        }
