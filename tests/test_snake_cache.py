import numpy as np

from slither_gym.rl.snake_cache import SnakeCache
from slither_gym.rl.types import EnemySnakeInfo


def _make_snake(snake_id: int, hx: float, hy: float) -> EnemySnakeInfo:
    return EnemySnakeInfo(
        snake_id=snake_id,
        head_x=hx, head_y=hy,
        mass=10.0, speed=3.0, angle=0.0, boosting=False,
        segments=np.array([[hx, hy]], dtype=np.float32),
    )


def test_empty_visible_returns_empty():
    cache = SnakeCache(max_slots=4)
    result = cache.update({}, 0.0, 0.0, 500.0)
    assert result == {}


def test_assigns_sequential_slots():
    cache = SnakeCache(max_slots=4)
    snakes = {1: _make_snake(1, 100, 0), 2: _make_snake(2, 200, 0)}
    result = cache.update(snakes, 0.0, 0.0, 500.0)
    assert set(result.keys()) == {1, 2}
    assert set(result.values()) <= {0, 1, 2, 3}


def test_stable_slots_across_updates():
    cache = SnakeCache(max_slots=4)
    snakes = {1: _make_snake(1, 100, 0), 2: _make_snake(2, 200, 0)}
    r1 = cache.update(snakes, 0.0, 0.0, 500.0)
    r2 = cache.update(snakes, 0.0, 0.0, 500.0)
    assert r1 == r2


def test_dead_snake_evicted():
    cache = SnakeCache(max_slots=4)
    snakes = {1: _make_snake(1, 100, 0), 2: _make_snake(2, 200, 0)}
    cache.update(snakes, 0.0, 0.0, 500.0)
    # Snake 1 disappears (died)
    snakes_after = {2: _make_snake(2, 200, 0)}
    result = cache.update(snakes_after, 0.0, 0.0, 500.0)
    assert 1 not in result
    assert 2 in result


def test_distant_snake_evicted():
    cache = SnakeCache(max_slots=4)
    snakes = {1: _make_snake(1, 100, 0)}
    cache.update(snakes, 0.0, 0.0, 500.0)
    # Snake 1 moves beyond 2x perception (>1000)
    snakes_far = {1: _make_snake(1, 1100, 0)}
    result = cache.update(snakes_far, 0.0, 0.0, 500.0)
    assert 1 not in result


def test_buffer_zone_keeps_snake():
    """Snake at 1.5x perception is kept (< 2x threshold)."""
    cache = SnakeCache(max_slots=4)
    snakes = {1: _make_snake(1, 100, 0)}
    cache.update(snakes, 0.0, 0.0, 500.0)
    # Snake at 750 = 1.5x perception — should stay
    snakes_mid = {1: _make_snake(1, 750, 0)}
    result = cache.update(snakes_mid, 0.0, 0.0, 500.0)
    assert 1 in result


def test_capacity_overflow_evicts_furthest():
    cache = SnakeCache(max_slots=2)
    snakes = {1: _make_snake(1, 100, 0), 2: _make_snake(2, 400, 0)}
    cache.update(snakes, 0.0, 0.0, 500.0)
    # New snake 3 is closer than snake 2 — should evict snake 2
    snakes_new = {
        1: _make_snake(1, 100, 0),
        2: _make_snake(2, 400, 0),
        3: _make_snake(3, 50, 0),
    }
    result = cache.update(snakes_new, 0.0, 0.0, 500.0)
    assert 1 in result
    assert 3 in result
    assert 2 not in result


def test_no_eviction_when_new_is_farther():
    cache = SnakeCache(max_slots=2)
    snakes = {1: _make_snake(1, 100, 0), 2: _make_snake(2, 200, 0)}
    cache.update(snakes, 0.0, 0.0, 500.0)
    # New snake 3 is farther than both — no eviction
    snakes_new = {
        1: _make_snake(1, 100, 0),
        2: _make_snake(2, 200, 0),
        3: _make_snake(3, 300, 0),
    }
    result = cache.update(snakes_new, 0.0, 0.0, 500.0)
    assert 3 not in result
    assert 1 in result and 2 in result


def test_reset_clears_all():
    cache = SnakeCache(max_slots=4)
    snakes = {1: _make_snake(1, 100, 0)}
    cache.update(snakes, 0.0, 0.0, 500.0)
    cache.reset()
    result = cache.update({}, 0.0, 0.0, 500.0)
    assert result == {}


def test_freed_slot_reused():
    cache = SnakeCache(max_slots=2)
    snakes = {1: _make_snake(1, 100, 0), 2: _make_snake(2, 200, 0)}
    r1 = cache.update(snakes, 0.0, 0.0, 500.0)
    slot_of_1 = r1[1]
    # Snake 1 dies, snake 3 appears
    snakes2 = {2: _make_snake(2, 200, 0), 3: _make_snake(3, 150, 0)}
    r2 = cache.update(snakes2, 0.0, 0.0, 500.0)
    assert 3 in r2
    # Snake 3 should get the freed slot
    assert r2[3] == slot_of_1


def test_16_slots_full():
    cache = SnakeCache(max_slots=16)
    snakes = {i: _make_snake(i, float(i * 10), 0.0) for i in range(16)}
    result = cache.update(snakes, 0.0, 0.0, 500.0)
    assert len(result) == 16
    assert set(result.values()) == set(range(16))
