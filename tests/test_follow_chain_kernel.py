"""T19 co-primary gate: the numba follow-the-leader kernel is BITWISE identical to the
reference python loop (PROPOSAL-T19-env-step-kernel §3) over 10k randomized chain updates
spanning realistic and adversarial geometry (zero-distance pairs, huge gaps, denormals-ish
tiny offsets, long chains)."""
import math

import numpy as np
import pytest

from slither_gym.core import snake as snake_mod


@pytest.mark.skipif(snake_mod.PHYSICS_KERNEL == "python", reason="numba not installed")
def test_kernel_bitwise_identical_10k_updates() -> None:
    rng = np.random.default_rng(0)
    checked = 0
    for case in range(10_000):
        n = int(rng.integers(2, 120))
        spacing = float(rng.uniform(0.5, 12.0))
        base = rng.uniform(-30000, 30000, size=2).astype(np.float32)
        # random walk chain with a mix of sub- and super-spacing gaps
        steps = rng.uniform(-2.5 * spacing, 2.5 * spacing, size=(n, 2)).astype(np.float32)
        if case % 7 == 0:
            steps[rng.integers(1, n)] = 0.0          # coincident pair -> d == 0 branch
        if case % 11 == 0:
            steps[rng.integers(1, n)] *= 1e4         # huge gap
        if case % 13 == 0:
            steps[rng.integers(1, n)] *= 1e-6        # tiny offset
        seg = (base + np.cumsum(steps, axis=0)).astype(np.float32)
        a = np.ascontiguousarray(seg)
        b = a.copy()
        snake_mod._follow_chain_py(a, n, spacing)
        snake_mod._follow_chain_numba(b, n, spacing)
        assert a.tobytes() == b.tobytes(), f"case {case}: kernel diverged from reference"
        checked += 1
    assert checked == 10_000


@pytest.mark.skipif(snake_mod.PHYSICS_KERNEL == "python", reason="numba not installed")
def test_kernel_zero_distance_no_nan() -> None:
    seg = np.zeros((4, 2), dtype=np.float32)
    snake_mod._follow_chain_numba(seg, 4, 5.0)
    assert not np.isnan(seg).any()
    ref = np.zeros((4, 2), dtype=np.float32)
    snake_mod._follow_chain_py(ref, 4, 5.0)
    assert seg.tobytes() == ref.tobytes()


def test_env_var_forces_python() -> None:
    # the dispatch honors SLITHER_PHYSICS_KERNEL=python at import; here just assert the
    # module exposes both paths and the active one is recorded.
    assert hasattr(snake_mod, "_follow_chain_py")
    assert snake_mod.PHYSICS_KERNEL in ("python", "numba-follow-chain-v1")
    assert (snake_mod.PHYSICS_KERNEL == "python") == (
        snake_mod._follow_chain is snake_mod._follow_chain_py)
