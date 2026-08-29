"""S4: action-latency randomization tests.

Four claims:
  1. Legacy zero-delay is BYTE-IDENTICAL to pre-S4 behavior, proven against
     golden fingerprints captured on the pre-S4 code (commit 8967a70).
  2. delay=k shifts the action's effect by exactly k physics ticks: the
     delayed run's heading/boost trajectory equals the undelayed run's,
     time-shifted by k, given constant initial conditions.
  3. Per-episode DR sampling draws integer delays uniformly and INCLUSIVELY
     from (min, max), and the realistic preset ships point=1 with band (0,3).
  4. The FIFO re-seeds on reset: stale pending actions from a previous
     episode never leak into the next one.
"""

import dataclasses
import hashlib
import math

import numpy as np

from slither_gym.core.realism import realistic_world_config, sample_world_config
from slither_gym.core.types import WorldConfig
from slither_gym.rl.action_delay import ActionDelayQueue
from slither_gym.rl.env_gym import SlitherGymEnv
from slither_gym.rl.env_parallel import SlitherParallelEnv

# --------------------------------------------------------------------------
# 1. Legacy (delay 0) byte-identity
# --------------------------------------------------------------------------

# Golden fingerprints originally captured on the pre-S4 code (commit 8967a70),
# before any action-delay plumbing existed:
#   gym      c36cbfc2c58325b4ca0da67eafcc5ff4129b472e5f7fc74be53eeb32765b70d7
#   parallel 4b328c2417b9dd2ace8a906af9820e083b0151b6cce0db98908f32b80b4dbc84
# Re-captured 2026-08-09 for the E32 pre-launch reward fix (rl/reward.py:
# remains_eaten double-pay removed; kill_reward_coef wired from config with an
# unchanged default of 5.0). These fingerprints fold the REWARD stream in, so
# they legitimately move with that fix. The state-only trajectory (head_x/y,
# angle, mass — everything else these hashes cover) was verified bitwise
# IDENTICAL before and after the reward change:
#   gym      state-only f2e2b3f8d3bb70814a0cedb0f530a19d8364a5cb35c336f78927a3441ef205e6
#   parallel state-only c97f8204ba11d6a05b78d85cd08a596c8a43209af34b96723e7598b404ea29dc
# so the S4 zero-delay byte-identity claim (physics/trajectory) still holds.
_GOLDEN_GYM = "70a328ab8fe782a3bc6cace3b1a589d2f3f0547283622e88839aaa407308e464"
_GOLDEN_PARALLEL = "fc62f721bd9762b5ae2c119c139084f8b11660965977ab16c58d8f9df2f76872"


def _gym_fingerprint(cfg: WorldConfig) -> str:
    env = SlitherGymEnv(cfg, num_bots=2, max_ticks=4000, seed=123)
    env.reset(seed=123)
    arng = np.random.default_rng(42)
    h = hashlib.sha256()
    for _ in range(60):
        a = np.array(
            [arng.uniform(-1, 1), arng.uniform(-1, 1), arng.uniform(0, 1)],
            dtype=np.float32,
        )
        _, r, term, trunc, info = env.step(a)
        s = info.get("snake_state")
        if s is not None:
            h.update(
                np.array(
                    [s.head_x, s.head_y, s.angle, s.mass], dtype=np.float64
                ).tobytes()
            )
        h.update(np.float64(r).tobytes())
        if term or trunc:
            break
    return h.hexdigest()


def _parallel_fingerprint(cfg: WorldConfig) -> str:
    env = SlitherParallelEnv(cfg, num_agents=3, seed=7)
    env.reset(seed=7)
    arng = np.random.default_rng(11)
    h = hashlib.sha256()
    for _ in range(60):
        actions = {
            agent: np.array(
                [arng.uniform(-1, 1), arng.uniform(-1, 1), arng.uniform(0, 1)],
                dtype=np.float32,
            )
            for agent in env.agents
        }
        _, rewards, _, _, _ = env.step(actions)
        for agent in sorted(rewards):
            h.update(agent.encode())
            h.update(np.float64(rewards[agent]).tobytes())
        assert env._world is not None
        for _, s in sorted(env._world.get_snake_states().items()):
            if s.alive:
                h.update(
                    np.array(
                        [s.head_x, s.head_y, s.angle, s.mass], dtype=np.float64
                    ).tobytes()
                )
        if not env.agents:
            break
    return h.hexdigest()


def test_s4_legacy_gym_env_byte_identical() -> None:
    cfg = WorldConfig()
    assert cfg.action_delay_ticks == 0
    assert _gym_fingerprint(cfg) == _GOLDEN_GYM


def test_s4_legacy_parallel_env_byte_identical() -> None:
    cfg = WorldConfig()
    assert _parallel_fingerprint(cfg) == _GOLDEN_PARALLEL


def test_s4_explicit_zero_delay_matches_golden_too() -> None:
    cfg = dataclasses.replace(
        WorldConfig(), action_delay_ticks=0,
        action_delay_ticks_min=0, action_delay_ticks_max=3,
    )
    # Ranges are inert while randomize_physics is False.
    assert _gym_fingerprint(cfg) == _GOLDEN_GYM


# --------------------------------------------------------------------------
# ActionDelayQueue unit semantics
# --------------------------------------------------------------------------

def test_s4_queue_fifo_semantics() -> None:
    q = ActionDelayQueue(2)
    q.seed(0.0)  # heading 0 -> (1, 0, False)
    a1 = (0.5, 0.5, True)
    a2 = (0.6, 0.6, False)
    a3 = (0.7, 0.7, True)
    assert q.apply(a1) == (1.0, 0.0, False)  # seeded straight
    assert q.apply(a2) == (1.0, 0.0, False)  # seeded straight
    assert q.apply(a3) == a1                  # commanded 2 ticks ago
    assert q.apply((0.0, 1.0, False)) == a2


def test_s4_queue_zero_delay_is_identity() -> None:
    q = ActionDelayQueue(0)
    q.seed(1.23)
    a = (0.1, 0.9, True)
    assert q.apply(a) is a  # the very same object, no repacking


# --------------------------------------------------------------------------
# 2. delay=k shifts the action's effect by exactly k ticks
# --------------------------------------------------------------------------

def _shift_rollout(delay: int, n_steps: int = 40, straight_steps: int = 6):
    """step_mul=1 so one RL step == one physics tick; no bots and (near) no
    food so the trajectory is purely action-driven. Command straight-ahead
    for `straight_steps`, then a hard turn + boost forever. Returns post-step
    (angles, boosting, positions, masses)."""
    cfg = dataclasses.replace(
        WorldConfig(),
        step_mul=1,
        food_spawn_rate=0,
        max_food=2,  # one far-away pellet at reset; nothing ever respawns
        action_delay_ticks=delay,
    )
    env = SlitherGymEnv(cfg, num_bots=0, max_ticks=10_000, seed=5)
    env.reset(seed=5)
    assert env._world is not None
    h0 = env._world.get_snake_states()[0].angle
    turn_target = h0 + 2.5  # far outside one tick's turn clamp

    angles, boosting, positions, masses = [], [], [], []
    for t in range(n_steps):
        if t < straight_steps:
            a = np.array([math.cos(h0), math.sin(h0), 0.0], dtype=np.float32)
        else:
            a = np.array(
                [math.cos(turn_target), math.sin(turn_target), 1.0],
                dtype=np.float32,
            )
        _, _, term, trunc, info = env.step(a)
        assert not term and not trunc
        s = info["snake_state"]
        angles.append(s.angle)
        boosting.append(s.boosting)
        positions.append((s.head_x, s.head_y))
        masses.append(s.mass)
    return h0, angles, boosting, positions, masses


def test_s4_delay_k_shifts_effect_by_exactly_k_ticks() -> None:
    n = 40
    h0_u, ang_u, boost_u, pos_u, mass_u = _shift_rollout(0, n)
    for k in (1, 2, 3):
        h0_d, ang_d, boost_d, pos_d, mass_d = _shift_rollout(k, n)
        assert h0_d == h0_u  # same seed -> same spawn heading
        m = n - k
        # Heading and boost trajectories: EXACT time-shift by k ticks.
        assert ang_d[k:] == ang_u[:m], f"delay={k}: heading not shifted by k"
        assert boost_d[k:] == boost_u[:m], f"delay={k}: boost not shifted by k"
        assert mass_d[k:] == mass_u[:m], f"delay={k}: mass not shifted by k"
        # During the first k ticks the delayed snake holds its spawn heading.
        for t in range(k):
            assert abs(ang_d[t] - h0_d) < 1e-9
            assert not boost_d[t]
        # Positions: the delayed run is the undelayed one displaced by the k
        # extra straight ticks flown at spawn -- a CONSTANT offset thereafter
        # (constant up to float summation order).
        offs = np.array(pos_d[k:]) - np.array(pos_u[:m])
        assert np.max(np.abs(offs - offs[0])) < 1e-6, f"delay={k}: offset drifts"
        expected = k * 3.0  # k ticks at legacy base_speed 3.0 u/tick
        assert abs(float(np.hypot(*offs[0])) - expected) < 1e-6


def test_s4_parallel_env_delay_holds_spawn_heading() -> None:
    cfg = dataclasses.replace(
        WorldConfig(), step_mul=1, food_spawn_rate=0, max_food=2,
        action_delay_ticks=2,
    )
    env = SlitherParallelEnv(cfg, num_agents=1, seed=3)
    env.reset(seed=3)
    assert env._world is not None
    h0 = env._world.get_snake_states()[0].angle
    turn = np.array(
        [math.cos(h0 + 2.5), math.sin(h0 + 2.5), 0.0], dtype=np.float32
    )
    for t in range(4):
        env.step({"snake_0": turn})
        angle = env._world.get_snake_states()[0].angle
        if t < 2:
            assert abs(angle - h0) < 1e-9, f"turn leaked through at tick {t}"
        else:
            assert abs(angle - h0) > 0.05, f"turn never arrived at tick {t}"


# --------------------------------------------------------------------------
# 3. Per-episode DR sampling
# --------------------------------------------------------------------------

def test_s4_realistic_preset_point_and_band() -> None:
    cfg = realistic_world_config()
    assert cfg.action_delay_ticks == 1
    assert cfg.action_delay_ticks_min == 0
    assert cfg.action_delay_ticks_max == 3
    assert cfg.randomize_physics is False


def test_s4_sample_world_config_draws_integers_inclusive() -> None:
    cfg = dataclasses.replace(
        WorldConfig(),
        randomize_physics=True,
        action_delay_ticks_min=0,
        action_delay_ticks_max=3,
    )
    seen: set[int] = set()
    for i in range(300):
        s = sample_world_config(cfg, np.random.default_rng(i))
        assert isinstance(s.action_delay_ticks, int)
        seen.add(s.action_delay_ticks)
    # Inclusive on BOTH ends: (0, 3) must cover {0, 1, 2, 3} and nothing else.
    assert seen == {0, 1, 2, 3}


def test_s4_sampling_off_draws_nothing() -> None:
    cfg = dataclasses.replace(
        WorldConfig(), action_delay_ticks_min=0, action_delay_ticks_max=3,
    )
    assert cfg.randomize_physics is False
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    assert sample_world_config(cfg, rng) is cfg
    assert rng.bit_generator.state == before  # zero RNG consumed


def test_s4_env_resamples_delay_per_episode() -> None:
    cfg = dataclasses.replace(
        WorldConfig(),
        randomize_physics=True,
        action_delay_ticks_min=0,
        action_delay_ticks_max=3,
    )
    env = SlitherGymEnv(cfg, num_bots=0, seed=0)
    seen: set[int] = set()
    for _ in range(60):
        env.reset()
        d = env._action_delay.delay_ticks
        assert d == env._world_config.action_delay_ticks  # episode-constant
        assert 0 <= d <= 3
        seen.add(d)
    assert seen == {0, 1, 2, 3}


# --------------------------------------------------------------------------
# 4. Queue reset on new episode
# --------------------------------------------------------------------------

def test_s4_queue_reseeds_on_reset() -> None:
    cfg = dataclasses.replace(
        WorldConfig(), step_mul=1, food_spawn_rate=0, max_food=2,
        action_delay_ticks=3,
    )
    env = SlitherGymEnv(cfg, num_bots=0, seed=9)

    def run_episode(n: int) -> list[float]:
        env.reset(seed=9)
        assert env._world is not None
        h0 = env._world.get_snake_states()[0].angle
        turn = np.array(
            [math.cos(h0 + 2.5), math.sin(h0 + 2.5), 1.0], dtype=np.float32
        )
        angles = []
        for _ in range(n):
            _, _, term, trunc, info = env.step(turn)
            assert not term and not trunc
            angles.append(info["snake_state"].angle)
        return [h0, *angles]

    first = run_episode(20)  # fills the FIFO with hard-turn/boost actions
    second = run_episode(20)  # reset must discard them
    # Identical episodes: the second reset re-seeded the queue (same spawn,
    # same actions, same trajectory). Stale pending turns would desync tick 0.
    assert first == second
    # And the seeded queue holds the spawn heading for exactly delay=3 ticks.
    h0 = first[0]
    for t in range(1, 4):
        assert abs(first[t] - h0) < 1e-9
    assert abs(first[4] - h0) > 0.05
