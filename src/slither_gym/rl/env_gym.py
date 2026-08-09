from __future__ import annotations

import math
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

from slither_gym.core.realism import sample_world_config
from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World
from slither_gym.obs.schema_v5 import ObsConfigV5
from slither_gym.rl.action_delay import ActionDelayQueue
from slither_gym.rl.bot_policy import BotPolicy
from slither_gym.rl.env_obs_v5 import (
    LastAction,
    v5_empty_obs,
    v5_initial_last_action,
    v5_observe,
)
from slither_gym.rl.env_parallel import SlitherParallelEnv
from slither_gym.rl.obs_processor import compute_observation
from slither_gym.rl.reward import compute_reward
from slither_gym.rl.snake_cache import SnakeCache
from slither_gym.rl.types import AgentId, EnemySnakeInfo, ObsConfig, RawGameState


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
        bot_policies: dict[int, Any] | None = None,
        obs_schema: str = "v4",
        obs_config_v5: ObsConfigV5 | None = None,
        bot_sct_law: str | None = None,
        bot_sct_log_median: float = 22.0,
        bot_sct_log_sigma: float = 1.632,
        bot_sct_min: float = 2.0,
        bot_sct_max: float = 256.0,
    ) -> None:
        super().__init__()
        if obs_schema not in ("v4", "v5"):
            raise ValueError(f"obs_schema must be 'v4' or 'v5', got {obs_schema!r}")
        # S5: opt-in deployable obs for the RL AGENT ONLY. Bots always stay on
        # the V4 obs path — scripted BotPolicy and every frozen self-play
        # checkpoint consume V4 observations; they model other players, not
        # the deployable agent.
        self._obs_schema = obs_schema
        self._obs_config_v5 = obs_config_v5 or ObsConfigV5()
        self._last_commanded_rl: LastAction = (1.0, 0.0, 0.0)
        # The pristine config. `_world_config` is the per-episode RESOLVED one
        # (identical to this unless world_config.randomize_physics is set);
        # sampling always starts from the base so jitter cannot compound.
        self._base_world_config = world_config
        self._world_config = world_config
        self._obs_config = obs_config
        self._num_bots = num_bots
        self._max_ticks = max_ticks
        self._seed = seed
        self._respawn_bots = respawn_bots
        self._render_mode = render_mode

        # P0.5 (frozen_eval_v8): opponent SIZE distribution. Real slither.io
        # opponents span sct 2-262 (measured p10 2, p50 22, p90 178 — 8-game
        # consolidated set, docs/SIM_REALISM_STATE.md), while the legacy sim
        # spawns every bot at initial_mass. When bot_sct_law == "lognormal",
        # each BOT (never the RL agent, snake 0) spawns AND respawns with
        #   sct ~ clip(lognormal(median=bot_sct_log_median,
        #                        sigma=bot_sct_log_sigma),
        #              bot_sct_min, bot_sct_max)
        # converted to mass via mass = initial_mass + (sct - initial_segments)
        # (the inverse of snake.py's sct formula). None (default) draws NO RNG
        # and spawns at initial_mass — byte-identical to every pre-P0.5 run.
        if bot_sct_law not in (None, "lognormal"):
            raise ValueError(
                f"bot_sct_law must be None or 'lognormal', got {bot_sct_law!r}"
            )
        self._bot_sct_law = bot_sct_law
        self._bot_sct_log_median = float(bot_sct_log_median)
        self._bot_sct_log_sigma = float(bot_sct_log_sigma)
        self._bot_sct_min = float(bot_sct_min)
        self._bot_sct_max = float(bot_sct_max)

        self._world: World | None = None
        self._rng = np.random.default_rng(seed)
        # E11 curriculum: optional runtime bot-difficulty override (None → use world_config).
        # The trainer anneals this via set_bot_difficulty(); eval/static envs leave it None.
        self._bot_difficulty_override: float | None = None
        self._bot_policy = BotPolicy(
            world_config, self._rng, bot_difficulty=self._bot_difficulty_override
        )
        self._bot_policies: dict[int, Any] = bot_policies or {}
        self._rl_agent_id: AgentId = "snake_0"
        self._tick_count: int = 0
        self._obs_update_counter: int = 0

        self._parallel_env = SlitherParallelEnv(
            world_config=world_config,
            obs_config=obs_config,
            num_agents=1 + num_bots,
            max_ticks=max_ticks,
            seed=seed,
            obs_schema=obs_schema,
            obs_config_v5=obs_config_v5,
        )

        self.observation_space = self._parallel_env.observation_space(self._rl_agent_id)
        self.action_space = self._parallel_env.action_space(self._rl_agent_id)

        self._snake_cache = SnakeCache(max_slots=obs_config.k_enemies)
        self._bot_obs_cache: dict[int, dict[str, NDArray[np.float32]]] = {}
        # E29 per-snake obs: distinct-k_danger ObsConfigs for opponents trained at a different
        # danger width than the agent (built lazily in _obs_config_for_bot).
        self._per_kdanger_obs_config: dict[int, ObsConfig] = {}
        self._prev_phi: float = 0.0  # E13: cut-readiness potential Φ(s) from the previous RL step
        # S4: latency FIFO for the RL action path only (bots are never delayed
        # — they model other players whose latency is implicit in their
        # behavior). Rebuilt each reset from the RESOLVED per-episode config,
        # so a randomized delay is constant within an episode.
        self._action_delay = ActionDelayQueue(world_config.action_delay_ticks)

    def set_bot_difficulty(self, difficulty: float | None) -> None:
        """E11 curriculum hook: override bot difficulty for subsequent episodes.

        Takes effect on the next reset (when the shared BotPolicy is re-rolled). 1.0 =
        full realistic mix; lower = more "careless" (non-fleeing, killable) prey so the
        agent samples kills. None restores the WorldConfig value. The eval never calls
        this, so eval bots stay frozen at the config default.
        """
        self._bot_difficulty_override = None if difficulty is None else float(difficulty)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        # Resolve this episode's physics from the PRISTINE base config, so
        # randomization cannot compound across resets. Returns the same object
        # and draws no RNG unless randomize_physics is set, which keeps every
        # pre-existing run's RNG stream bit-identical.
        resolved = sample_world_config(self._base_world_config, self._rng)
        config_changed = resolved is not self._world_config
        self._world_config = resolved

        # Rebuilding BotPolicy consumes RNG, so only do it on the paths that
        # already did (a reseed) plus the new randomized-physics path.
        if seed is not None or config_changed:
            self._bot_policy = BotPolicy(
                self._world_config, self._rng, bot_difficulty=self._bot_difficulty_override
            )

        self._world = World(self._world_config, seed=self._seed)
        self._tick_count = 0
        self._snake_cache.reset()

        for i in range(1 + self._num_bots):
            self._world.spawn_snake(i, mass=self._bot_spawn_mass() if i > 0 else None)

        # S4: re-sample the per-episode delay (resolved config) and seed the
        # FIFO with the RL snake's spawn heading / no boost, so the first
        # `delay` ticks match an uncommanded snake exactly.
        self._action_delay = ActionDelayQueue(self._world_config.action_delay_ticks)
        spawn_angle = self._world.get_snake_states()[0].angle
        self._action_delay.seed(spawn_angle)
        # S5: last-commanded seeds with the spawn heading / no boost, exactly
        # like the delay queue.
        self._last_commanded_rl = v5_initial_last_action(spawn_angle)

        rl_obs = self._get_rl_observation()
        self._update_bot_obs_cache()
        # E13: seed Φ(s_0) so the first step's shaping telescopes correctly.
        self._prev_phi = (
            self._compute_cut_potential() if self._world_config.kill_shaping_coef != 0.0 else 0.0
        )
        return rl_obs, {}

    def _compute_cut_potential(self) -> float:
        """E13 cut-readiness potential Φ(s) ∈ [0, 1] — STATE-ONLY (no actions/time/history).

        Φ = max over alive enemies of [proximity · alignment], where for an enemy head h with
        unit heading ĥ and my nearest *body* segment b (segments behind my head):
          proximity = max(0, 1 − |h−b| / R)              (1 when touching, 0 beyond R)
          alignment = max(0, ĥ · unit(b − h))            (1 when charging straight at b, 0 if away)
        High Φ ⇔ an enemy is about to drive its head into my body (= a kill credited to me). Φ→0
        when no enemy is near/aligned. Kill geometry uses MY BODY, not my head (head-on = my death).
        """
        world = self._world
        if world is None:
            return 0.0
        R = float(self._world_config.kill_shaping_radius)
        # My alive body segments EXCLUDING the head region (a kill needs head→body, not head→head).
        owner = world._seg_owner
        alive = world._seg_alive
        mine = (owner == 0) & alive
        my_segs = world._segments[mine]
        if my_segs.shape[0] <= 1:
            return 0.0
        states = world.get_snake_states()
        me = states.get(0)
        if me is None or not me.alive:
            return 0.0
        # Drop the segment nearest my head so "proximity to my own head" isn't counted as cut-ready.
        head = np.array([me.head_x, me.head_y], dtype=np.float32)
        d_head = np.sum((my_segs - head) ** 2, axis=1)
        body = my_segs[d_head > (2.0 * self._world_config.segment_spacing) ** 2]
        if body.shape[0] == 0:
            return 0.0
        best = 0.0
        for sid, s in states.items():
            if sid == 0 or not s.alive:
                continue
            h = np.array([s.head_x, s.head_y], dtype=np.float32)
            rel = body - h                                   # vectors head→each of my segments
            d2 = np.sum(rel * rel, axis=1)
            j = int(np.argmin(d2))
            dist = float(np.sqrt(d2[j]))
            if dist >= R:
                continue
            proximity = 1.0 - dist / R
            seg_dir = rel[j] / (dist + 1e-6)                 # unit head→nearest segment
            align = math.cos(s.angle) * float(seg_dir[0]) + math.sin(s.angle) * float(seg_dir[1])
            phi = proximity * max(0.0, align)
            if phi > best:
                best = phi
        return best

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
        # S5: record the COMMANDED (pre-delay-queue) action; the v5 obs feeds
        # it back as self_state[7] and [9:12] next step (env_obs_v5 docstring).
        self._last_commanded_rl = (cos_a, sin_a, 1.0 if boost else 0.0)

        bot_actions: dict[int, tuple[float, float, bool]] = {}
        # Group alive, obs-cached bots by their policy object so policies that support batched
        # inference (self-play/league ActorCriticBotPolicy) run ONE forward for all their bots
        # instead of one per bot (the ~3.5× self-play speedup). Bots grouped by identity, so a
        # league with distinct opponent policies batches each policy's bots separately. Scripted
        # BotPolicy (no act_batch) falls through to the per-bot path → byte-identical behavior.
        policy_groups: dict[int, tuple[Any, list[int]]] = {}
        for i in range(1, 1 + self._num_bots):
            state = self._world.get_snake_states().get(i)
            if state is not None and state.alive:
                if i in self._bot_obs_cache:
                    policy = self._bot_policies.get(i, self._bot_policy)
                    policy_groups.setdefault(id(policy), (policy, []))[1].append(i)
                else:
                    bot_actions[i] = (math.cos(state.angle), math.sin(state.angle), False)
        for policy, ids in policy_groups.values():
            if len(ids) > 1 and hasattr(policy, "act_batch"):
                acts = policy.act_batch([self._bot_obs_cache[i] for i in ids])
                for i, a in zip(ids, acts):
                    bot_actions[i] = (float(a[0]), float(a[1]), bool(a[2] > 0.5))
            else:
                for i in ids:
                    a = policy.act(self._bot_obs_cache[i], snake_id=i)
                    bot_actions[i] = (float(a[0]), float(a[1]), bool(a[2] > 0.5))

        total_reward = 0.0
        terminated = False
        last_step_result = None

        for _ in range(config.step_mul):
            # S4: per-tick FIFO — the action applied at tick t is the one
            # commanded at t - delay. No-op (returns the tuple unchanged)
            # when action_delay_ticks == 0, the legacy default.
            world_actions: dict[int, tuple[float, float, bool]] = {
                0: self._action_delay.apply((cos_a, sin_a, boost))
            }
            world_actions.update(bot_actions)

            results = self._world.step(world_actions)
            self._tick_count += 1

            if 0 in results:
                result = results[0]
                last_step_result = result
                snake_state = self._world.get_snake_states().get(0)
                if snake_state is not None:
                    reward = compute_reward(result, snake_state, config)
                    total_reward += reward
                if not result.alive:
                    terminated = True
                    break

        truncated = not terminated and self._tick_count >= self._max_ticks

        # E13: potential-based kill-credit shaping, applied ONCE per RL step (here, the RL-step
        # boundary), not per physics tick. r_shape = coef·(γ·Φ(s′) − Φ(s)). Terminal Φ(s′)=0.
        # Computed before respawn so Φ reflects the state the agent actually acted into.
        if config.kill_shaping_coef != 0.0:
            phi_next = 0.0 if terminated else self._compute_cut_potential()
            total_reward += config.kill_shaping_coef * (
                config.kill_shaping_gamma * phi_next - self._prev_phi
            )
            self._prev_phi = phi_next

        if self._respawn_bots:
            for i in range(1, 1 + self._num_bots):
                state = self._world.get_snake_states().get(i)
                if state is None or not state.alive:
                    # P0.5: respawns re-draw from the same size law, keeping the
                    # opponent size DISTRIBUTION stationary over the episode.
                    self._world.spawn_snake(i, mass=self._bot_spawn_mass())
                    # Reset stateful policies on respawn
                    if i in self._bot_policies and hasattr(self._bot_policies[i], 'reset'):
                        self._bot_policies[i].reset(i)

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
            info["snake_state"] = rl_state

        # Expose StepResult and WorldConfig for external reward computation
        if last_step_result is not None:
            info["step_result"] = last_step_result
            if terminated:
                # Death cause from killed_by: None = ran out of bounds (wall); otherwise
                # = ran into snake <id>'s body. Self-collision is impossible (the collision
                # query excludes own segments), so those are the only two causes.
                kb = last_step_result.killed_by
                info["death_cause"] = "wall" if kb is None else f"enemy:{kb}"
        info["world_config"] = self._world_config

        return rl_obs, total_reward, terminated, truncated, info

    def _get_rl_observation(self) -> dict[str, NDArray[np.float32]]:
        assert self._world is not None
        states = self._world.get_snake_states()
        rl_state = states.get(0)
        if rl_state is None or not rl_state.alive:
            return self._empty_obs()

        # S5: the deployable obs path (visibility mask + canonical build_obs).
        if self._obs_schema == "v5":
            return v5_observe(
                self._world, 0, self._obs_config_v5, self._last_commanded_rl
            )

        food_pos = self._world.get_food_positions()
        food_vals = self._world.get_food_values()
        raw = self._build_raw_state(0, rl_state, states, food_pos, food_vals)

        # Update snake cache and get slot mapping for RL agent
        visible = {info.snake_id: info for info in raw.enemy_snakes}
        slot_mapping = self._snake_cache.update(
            visible, rl_state.head_x, rl_state.head_y,
            self._world_config.perception_radius,
        )
        return compute_observation(raw, self._obs_config, snake_slot_mapping=slot_mapping)

    def _bot_spawn_mass(self) -> float | None:
        """Sample a bot's spawn mass from the configured size law.

        Returns None (spawn at initial_mass, zero RNG draws — legacy
        byte-identity) unless bot_sct_law is set. "lognormal": sct is drawn
        from lognormal(ln(median), sigma) and clipped to
        [bot_sct_min, bot_sct_max]; the sim's segment cap
        (max_segments_per_snake, 256) is the natural ceiling. Draws exactly
        one value from self._rng per call, so pinned seeds stay deterministic.
        """
        if self._bot_sct_law is None:
            return None
        sct = float(
            self._rng.lognormal(
                mean=math.log(self._bot_sct_log_median), sigma=self._bot_sct_log_sigma
            )
        )
        sct = min(max(sct, self._bot_sct_min), self._bot_sct_max)
        cfg = self._world_config
        # Inverse of snake.py: sct = initial_segments + (mass - initial_mass).
        return cfg.initial_mass + (sct - cfg.initial_segments)

    def _update_bot_obs_cache(self) -> None:
        assert self._world is not None
        self._bot_obs_cache.clear()
        states = self._world.get_snake_states()
        food_pos = self._world.get_food_positions()
        food_vals = self._world.get_food_values()
        food_corpse = self._world.get_food_is_corpse()

        for i in range(1, 1 + self._num_bots):
            bot_state = states.get(i)
            if bot_state is None or not bot_state.alive:
                continue
            raw = self._build_raw_state(i, bot_state, states, food_pos, food_vals, food_corpse)
            self._bot_obs_cache[i] = compute_observation(raw, self._obs_config_for_bot(i))

    def _obs_config_for_bot(self, i: int) -> ObsConfig:
        """Per-snake obs (E29 refactor): a policy-opponent trained at a different danger width
        must receive obs at ITS width, not the (possibly wider) agent's. Falls back to the global
        config for scripted bots / same-width policies. Cached per distinct k_danger."""
        policy = self._bot_policies.get(i, self._bot_policy)
        k = int(getattr(policy, "k_danger_segments", self._obs_config.k_danger_segments))
        if k == self._obs_config.k_danger_segments:
            return self._obs_config
        cache = self._per_kdanger_obs_config
        if k not in cache:
            import dataclasses
            cache[k] = dataclasses.replace(self._obs_config, k_danger_segments=k)
        return cache[k]

    def _build_raw_state(
        self,
        snake_id: int,
        state: Any,
        all_states: dict[int, Any],
        food_pos: NDArray[np.float32] | None = None,
        food_vals: NDArray[np.float32] | None = None,
        food_is_corpse: NDArray[np.bool_] | None = None,
    ) -> RawGameState:
        assert self._world is not None
        if food_pos is None:
            food_pos = self._world.get_food_positions()
        if food_vals is None:
            food_vals = self._world.get_food_values()
        if food_is_corpse is None:
            food_is_corpse = self._world.get_food_is_corpse()

        # Own body segments
        own_segs = self._world.get_segments(snake_id)

        enemy_segs_list: list[NDArray[np.float32]] = []
        enemy_radius_list: list[float] = []
        enemy_snakes_list: list[EnemySnakeInfo] = []

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

            # Flat segments for danger_segments
            enemy_segs_list.append(segs)
            n = len(segs)
            enemy_radius_list.extend([other_state.segment_radius] * n)

            # Per-snake structured data for enemies channel
            enemy_snakes_list.append(EnemySnakeInfo(
                snake_id=other_id,
                head_x=other_state.head_x,
                head_y=other_state.head_y,
                mass=other_state.mass,
                speed=other_state.speed,
                angle=other_state.angle,
                boosting=other_state.boosting,
                segments=segs,
            ))

        if enemy_segs_list:
            all_enemy_segs: NDArray[np.float32] = np.concatenate(enemy_segs_list, axis=0)
            all_radius = np.array(enemy_radius_list, dtype=np.float32)
        else:
            all_enemy_segs = np.zeros((0, 2), dtype=np.float32)
            all_radius = np.zeros(0, dtype=np.float32)

        if all_positions_list:
            all_positions = np.array(all_positions_list, dtype=np.float32)
            all_masses = np.array(all_masses_list, dtype=np.float32)
        else:
            all_positions = np.zeros((0, 2), dtype=np.float32)
            all_masses = np.zeros(0, dtype=np.float32)

        # Dummy arrays for legacy flat fields still needed by danger_segments
        n_flat = len(all_enemy_segs)
        return RawGameState(
            self_x=state.head_x,
            self_y=state.head_y,
            self_mass=state.mass,
            self_speed=state.speed,
            self_angle=state.angle,
            self_segment_count=state.segment_count,
            self_boosting=state.boosting,
            food_positions=food_pos,
            food_values=food_vals,
            food_is_corpse=food_is_corpse,
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

    def set_bot_policies(self, policies: dict[int, Any]) -> None:
        """Swap bot policies at runtime. Called by training loop for self-play."""
        self._bot_policies = policies

    def _empty_obs(self) -> dict[str, NDArray[np.float32]]:
        if self._obs_schema == "v5":
            return v5_empty_obs(self._obs_config_v5)
        obs_config = self._obs_config
        return {
            "self_state": np.zeros(8, dtype=np.float32),
            "food": np.zeros((obs_config.k_food, obs_config.food_features), dtype=np.float32),
            "prey": np.zeros((obs_config.k_prey, obs_config.prey_features), dtype=np.float32),
            "enemies": np.zeros((obs_config.k_enemies, obs_config.enemy_features), dtype=np.float32),
            "danger_segments": np.zeros((obs_config.k_danger_segments, obs_config.danger_features), dtype=np.float32),
            "own_body": np.zeros((obs_config.k_own_body, obs_config.own_body_features), dtype=np.float32),
            "minimap": np.zeros((obs_config.minimap_size, obs_config.minimap_size), dtype=np.float32),
        }
