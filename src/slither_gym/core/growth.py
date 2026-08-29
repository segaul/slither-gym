"""R3: the real client's growth law — total mass <-> (sct, fam) via fpsls/fmlts.

The real slither.io client ships two lookup tables in its config
(probe report.config, capture 2026-08-09T03-22-10):

    fmlts[b] = (1 - b/mscps)^2.25          for b < mscps, then held constant
    fpsls[0] = 0
    fpsls[b] = fpsls[b-1] + 1/fmlts[b-1]   for b <= mscps, then held constant

and the client's own size/score quantity ("length" in docs/REAL_GAME_DATA.md,
"real mass" here) is

    mass = (fpsls[sct] + fam/fmlts[sct] - 1) * 15 - 5

with fam in [0, 1) the fractional remainder toward the next segment (verified
hand-check, docs/captains-log/2026-W32.md: sct 4->5 with fam 0.9619->0.1171 is
a CONTINUOUS mass rise). The recurrence was verified BITWISE (max rel err 0.0
over all 2x2479 entries) against the game01 capture's report.config with
mscps=430, so this module is the single shared source of the law for the sim
core, the V5 obs schema, and the deployment bridge.

Why this matters (the R3 defect): the law is strongly SUPERLINEAR — one
segment costs ~16.9 mass at sct 22 and ~113 mass at sct 255, while a mean
pellet (value 6.25) is worth only ~1.3 mass. The legacy sim's
"sct = initial_segments + (mass - initial_mass)" with pellet value -> mass 1:1
grew segments ~79x too fast and saturated the sct-256 physics cap ~23 s into
an episode (derivation: docs/experiments/data/growth_law_r3.py, slither-rl).
Under WorldConfig.growth_law == "real" the sim tracks mass in this client
currency and derives (sct, fam) through the exact inverse below.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

CLIENT_MSCPS = 430  # probe report.config.mscps
CLIENT_LUT_LEN = 2479  # len(report.config.fmlts) == len(report.config.fpsls)

# The client's own minimum: real snakes spawn at sct 2, and the measured-exact
# body-scale law sc = 1 + (sct - 2)/106 is anchored there.
MIN_REAL_SCT = 2


def build_mass_luts(
    mscps: int = CLIENT_MSCPS, n: int = CLIENT_LUT_LEN
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce the client's (fpsls, fmlts) tables from its recurrence."""
    fmlts = np.empty(n, dtype=np.float64)
    fpsls = np.empty(n, dtype=np.float64)
    for b in range(n):
        fmlts[b] = (1.0 - b / mscps) ** 2.25 if b < mscps else fmlts[b - 1]
        if b == 0:
            fpsls[b] = 0.0
        elif b <= mscps:
            fpsls[b] = fpsls[b - 1] + 1.0 / fmlts[b - 1]
        else:
            fpsls[b] = fpsls[b - 1]
    return fpsls, fmlts


_DEFAULT_FPSLS, _DEFAULT_FMLTS = build_mass_luts()


def real_mass(
    sct: int,
    fam: float,
    fpsls: NDArray[np.float64] | None = None,
    fmlts: NDArray[np.float64] | None = None,
) -> float:
    """Real-client mass: (fpsls[sct] + fam/fmlts[sct] - 1) * 15 - 5.

    Hand-checked against the W32 fam-wrap entry (docs/captains-log/2026-W32.md):
    sct 4 -> 5 with fam 0.9619 -> 0.1171 is a continuous +0.155 gain in
    sct+fam units, and this formula maps it to a continuous mass rise.
    """
    if fpsls is None:
        fpsls = _DEFAULT_FPSLS
    if fmlts is None:
        fmlts = _DEFAULT_FMLTS
    i = min(max(int(sct), 0), len(fpsls) - 1)
    return (float(fpsls[i]) + fam / float(fmlts[i]) - 1.0) * 15.0 - 5.0


def real_mass_continuous(sct_continuous: float) -> float:
    """Mass at a continuous size s = sct + fam (fam = the fractional part)."""
    s = int(sct_continuous)
    return real_mass(s, sct_continuous - s)


def sct_fam_from_mass(mass: float) -> tuple[int, float]:
    """EXACT inverse of `real_mass` on the client LUT grid.

    Solving (fpsls[sct] + fam/fmlts[sct] - 1)*15 - 5 = mass for the unique
    (sct, fam) with fam in [0, 1): let Q = (mass + 5)/15 + 1; then sct is the
    largest index with fpsls[sct] <= Q, and fam = (Q - fpsls[sct])*fmlts[sct].
    fam < 1 holds automatically because fpsls[sct+1] = fpsls[sct] + 1/fmlts[sct].

    sct is floored at MIN_REAL_SCT (the client's spawn size; masses below
    real_mass(2, 0) = 10.079 clamp to sct 2, fam 0 — WorldConfig.initial_mass
    10.0 sits a hair under that on purpose so sim spawns match real spawns).
    """
    q = (mass + 5.0) / 15.0 + 1.0
    i = int(np.searchsorted(_DEFAULT_FPSLS, q, side="right")) - 1
    if i < MIN_REAL_SCT:
        return MIN_REAL_SCT, 0.0
    i = min(i, len(_DEFAULT_FPSLS) - 1)
    fam = (q - float(_DEFAULT_FPSLS[i])) * float(_DEFAULT_FMLTS[i])
    return i, min(max(fam, 0.0), 1.0 - 1e-12)
