import time

import numpy as np

from slither_gym.core.spatial_hash import SpatialHash


def test_basic_query() -> None:
    sh = SpatialHash(cell_size=50.0, bounds=3000.0)
    segments = np.array([[0, 0], [100, 0], [200, 0]], dtype=np.float32)
    alive = np.array([True, True, True], dtype=np.bool_)
    ids = np.array([0, 1, 2], dtype=np.int32)

    sh.rebuild(segments, alive, ids)
    result = sh.query_near(1.0, 0.0, 10.0, exclude_snake_id=-1)

    assert len(result) == 1
    assert result[0] == (0, 0)


def test_exclude_snake() -> None:
    sh = SpatialHash(cell_size=50.0, bounds=3000.0)
    segments = np.array([[0, 0], [5, 0]], dtype=np.float32)
    alive = np.array([True, True], dtype=np.bool_)
    ids = np.array([0, 1], dtype=np.int32)

    sh.rebuild(segments, alive, ids)
    result = sh.query_near(0.0, 0.0, 10.0, exclude_snake_id=0)

    assert len(result) == 1
    assert result[0][1] == 1  # only snake 1


def test_empty_query() -> None:
    sh = SpatialHash(cell_size=50.0, bounds=3000.0)
    segments = np.array([[1000, 1000]], dtype=np.float32)
    alive = np.array([True], dtype=np.bool_)
    ids = np.array([0], dtype=np.int32)

    sh.rebuild(segments, alive, ids)
    result = sh.query_near(0.0, 0.0, 10.0, exclude_snake_id=-1)
    assert len(result) == 0


def test_matches_brute_force() -> None:
    rng = np.random.default_rng(42)
    n = 1000
    segments = rng.uniform(-3000, 3000, (n, 2)).astype(np.float32)
    alive = np.ones(n, dtype=np.bool_)
    ids = np.arange(n, dtype=np.int32)

    sh = SpatialHash(cell_size=50.0, bounds=3000.0)
    sh.rebuild(segments, alive, ids)

    qx, qy = 100.0, 200.0
    radius = 30.0

    # Spatial hash result
    sh_result = set(sh.query_near(qx, qy, radius, exclude_snake_id=-1))

    # Brute force
    bf_result = set()
    for i in range(n):
        dx = float(segments[i, 0]) - qx
        dy = float(segments[i, 1]) - qy
        if dx * dx + dy * dy <= radius * radius:
            bf_result.add((i, i))

    assert sh_result == bf_result


def test_benchmark() -> None:
    rng = np.random.default_rng(42)
    n = 1000
    segments = rng.uniform(-3000, 3000, (n, 2)).astype(np.float32)
    alive = np.ones(n, dtype=np.bool_)
    ids = (np.arange(n, dtype=np.int32) % 10)

    sh = SpatialHash(cell_size=50.0, bounds=3000.0)

    # Warmup
    for _ in range(10):
        sh.rebuild(segments, alive, ids)
        for _ in range(10):
            sh.query_near(0.0, 0.0, 30.0, exclude_snake_id=0)

    iters = 200
    start = time.perf_counter()
    for _ in range(iters):
        sh.rebuild(segments, alive, ids)
        for _ in range(10):
            sh.query_near(
                float(rng.uniform(-3000, 3000)),
                float(rng.uniform(-3000, 3000)),
                30.0,
                exclude_snake_id=0,
            )
    elapsed = (time.perf_counter() - start) / iters * 1000

    assert elapsed < 0.4, f"rebuild + 10 queries too slow: {elapsed:.4f}ms (need <0.4ms)"
