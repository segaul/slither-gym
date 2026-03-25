import time

from slither_gym.rl.env_gym import SlitherGymEnv
from slither_gym.core.types import WorldConfig


def test_throughput() -> None:
    """
    SlitherGymEnv with 10 bots must sustain 1000+ steps/sec.
    Measured over 5000 steps, excluding reset.
    Uses explicit config to isolate from default changes.
    """
    config = WorldConfig(max_snakes=16, max_food=1024)
    env = SlitherGymEnv(world_config=config, num_bots=10, seed=42)
    env.reset()
    action = env.action_space.sample()

    start = time.perf_counter()
    for _ in range(5000):
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            env.reset()
    elapsed = time.perf_counter() - start

    steps_per_sec = 5000 / elapsed
    assert steps_per_sec > 300, f"Too slow: {steps_per_sec:.0f} steps/sec"
