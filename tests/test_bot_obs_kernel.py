"""INFRA-batched-hotspots gate (bot-obs half): the kernel's enemy rows are BITWISE the
reference per-bot loop's, checked live against a stepping env — every scripted bot's
cached obs each step equals a fresh reference rebuild (enemy_obs=None path) byte for
byte, across crowded geometry where culls, ties and the k_enemies truncation all fire."""
import numpy as np
import pytest

from slither_gym.rl import bot_obs_kernel
from slither_gym.rl.env_gym import SlitherGymEnv
from slither_gym.rl.types import ObsConfig
from slither_gym.core.types import WorldConfig


def _mk_env(seed: int, num_bots: int = 40) -> SlitherGymEnv:
    env = SlitherGymEnv(
        world_config=WorldConfig(map_radius=2000.0, max_snakes=num_bots + 1,
                                 max_food=4096),
        num_bots=num_bots,
        max_ticks=2000,
        seed=seed,
    )
    env.reset(seed=seed)
    return env


@pytest.mark.skipif(bot_obs_kernel.BOT_OBS_KERNEL == "python",
                    reason="numba not installed")
def test_kernel_rows_bitwise_vs_reference_live_env() -> None:
    rng = np.random.default_rng(0)
    env = _mk_env(seed=0)
    assert env._world is not None
    checked = 0
    for step in range(120):
        a = np.array([np.cos(rng.uniform(-np.pi, np.pi)),
                      np.sin(rng.uniform(-np.pi, np.pi)),
                      1.0 if rng.uniform() < 0.2 else 0.0], dtype=np.float32)
        _, _, term, trunc, _ = env.step(a)
        # Rebuild the cache on the settled post-step world so the reference rebuild
        # below sees the exact same snapshot the kernel path consumed.
        env._update_bot_obs_cache()
        states = env._world.get_snake_states()
        for i in sorted(env._bot_obs_partial):
            cached = env._bot_obs_cache[i]
            ref = env._compute_bot_observation_fast(
                i, states[i], states, enemy_obs=None)
            for key in cached:
                assert cached[key].tobytes() == ref[key].tobytes(), (
                    f"step {step} bot {i} field {key} diverged")
            checked += 1
        if term or trunc:
            env.reset(seed=step + 1)
    assert checked > 500, "gate exercised too few bot observations"


@pytest.mark.skipif(bot_obs_kernel.BOT_OBS_KERNEL == "python",
                    reason="numba not installed")
def test_shared_zero_arrays_are_locked() -> None:
    env = _mk_env(seed=3)
    env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    some_bot = next(iter(env._bot_obs_partial))
    obs = env._bot_obs_cache[some_bot]
    for key in ("prey", "danger_segments", "own_body", "minimap"):
        assert not obs[key].flags.writeable
        assert not obs[key].any()
        with pytest.raises(ValueError):
            obs[key][0] = 1.0
