import numpy as np

from slither_gym.rl.reward import compute_reward
from slither_gym.core.types import SnakeState, StepResult, WorldConfig


def _make_state(head_x: float = 0.0, head_y: float = 0.0) -> SnakeState:
    return SnakeState(
        snake_id=0, alive=True, mass=10.0, speed=3.0, angle=0.0,
        boosting=False, head_x=head_x, head_y=head_y,
        segment_count=10, segment_radius=3.0, turn_rate=0.1,
    )


def test_food_gain() -> None:
    config = WorldConfig()
    result = StepResult(alive=True, mass_delta=3.0, killed_by=None, kill_count=0, remains_eaten=0.0)
    r = compute_reward(result, _make_state(), config)
    np.testing.assert_allclose(r, 3.0 + 0.01, atol=1e-5)


def test_death_penalty() -> None:
    config = WorldConfig()
    result = StepResult(alive=False, mass_delta=0.0, killed_by=1, kill_count=0, remains_eaten=0.0)
    r = compute_reward(result, _make_state(), config)
    assert r < -9.0  # -10.0 + 0.01


def test_kill_bonus() -> None:
    config = WorldConfig()
    result = StepResult(alive=True, mass_delta=0.0, killed_by=None, kill_count=1, remains_eaten=0.0)
    r = compute_reward(result, _make_state(), config)
    np.testing.assert_allclose(r, 5.0 + 0.01, atol=1e-5)


def test_boundary_penalty() -> None:
    config = WorldConfig(map_radius=1000.0)
    # At 90% of map radius
    state = _make_state(head_x=900.0, head_y=0.0)
    result = StepResult(alive=True, mass_delta=0.0, killed_by=None, kill_count=0, remains_eaten=0.0)
    r = compute_reward(result, state, config)
    assert r < 0.01  # should have negative boundary penalty


def test_no_boundary_penalty_at_center() -> None:
    config = WorldConfig(map_radius=1000.0)
    state = _make_state(head_x=500.0, head_y=0.0)
    result = StepResult(alive=True, mass_delta=0.0, killed_by=None, kill_count=0, remains_eaten=0.0)
    r = compute_reward(result, state, config)
    np.testing.assert_allclose(r, 0.01, atol=1e-5)


def test_corpse_eating_bonus() -> None:
    config = WorldConfig()
    result = StepResult(alive=True, mass_delta=2.0, killed_by=None, kill_count=0, remains_eaten=2.0)
    r = compute_reward(result, _make_state(), config)
    # mass_delta * 1.0 + remains_eaten * 1.0 + survival = 2.0 + 2.0 + 0.01
    np.testing.assert_allclose(r, 4.01, atol=1e-5)
