"""State Space V5 — the canonical observation schema and builder.

Design contract (docs/STATE_SPACE_V5_PLAN.md, P1-P5):

- `VisibleState` contains ONLY fields a real browser client demonstrably
  delivers (probe-verified, 2026-08-09T03-22-10 8-game session: 13,492 ego
  frames, 1,947 snapshots).
- `build_obs()` is a pure, deterministic, stateless function of a
  `VisibleState`. It is the ONE implementation imported by both the sim
  (via `slither_gym.obs.visibility`) and the deployment bridge (S2).
- No sim-side statefulness: no slot persistence, no cross-frame memory.
  Layouts are order-canonical (stable nearest-first sorts).
- Every window/norm constant carries its probe citation and lives in
  `ObsConfigV5` — nothing is hard-coded in the builder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Client mass LUTs (fpsls / fmlts)
# ---------------------------------------------------------------------------

# The client's LUT recurrence (bitwise-verified vs the game01 capture's
# report.config, mscps=430) lives in core/growth.py so the sim's R3 growth law
# and this obs schema share ONE source; re-exported here for the bridge and
# for backwards compatibility of existing imports.
from slither_gym.core.growth import (  # noqa: F401  (re-exports)
    CLIENT_LUT_LEN,
    CLIENT_MSCPS,
    build_mass_luts,
    real_mass,
)


# ---------------------------------------------------------------------------
# VisibleState — only what the real browser client delivers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnemyVisible:
    """One enemy snake as the client delivers it.

    No fam for enemies (the client only sends own fam), so enemy mass is
    derived from sct alone. No boost flag either — the probe showed sp-based
    boost detection is unreliable at large sct (base sp rises with size);
    boost rides implicitly on `speed` (U-obs-2).
    """

    snake_id: int
    head_x: float  # world coords (client xx)
    head_y: float
    angle: float  # rad
    speed: float  # u/s
    sct: int
    # (N, 2) world coords, HEAD-FIRST (canonical order; the raw client array is
    # tail-first — the bridge/visibility layer reverses it). Possibly decimated
    # to ~64 u spacing, as the client delivers (median spacing 64.5 u).
    pts: NDArray[np.float32]


@dataclass(frozen=True)
class VisibleState:
    """Everything build_obs() is allowed to see. Probe-verified fields only."""

    # Own snake (client `snake` object + our own commanded action)
    head_x: float  # world coords (client xx)
    head_y: float
    angle: float  # rad (client ang)
    speed: float  # u/s (client sp converted; measured base 181.5 u/s)
    sct: int  # client sct (segment count)
    fam: float  # client fam (fractional mass remainder, wraps at sct+1)
    scale: float  # client sc (body scale, = 1 + (sct-2)/106 measured exact)
    boosting: bool  # from OUR OWN commanded action, not the sp>8 heuristic
    # (N, 2) world coords, head-first (client-delivered own pts, reversed to
    # canonical head-first order by the producer).
    body_pts: NDArray[np.float32]

    # Map geometry: the real world is centred at (grd, grd) with measured
    # radius 15000 (border hit at r=14976.5). The sim world is centred at
    # (0, 0) with its configured radius.
    map_center_x: float
    map_center_y: float
    map_radius: float

    # Other entities, already masked to the client delivery model.
    enemies: tuple[EnemyVisible, ...] = ()
    # (N, 3): x, y, value — value on the REAL scale (measured 3.0..14.2).
    # There is no is_corpse anywhere: corpse pellets are just big-value food
    # (P4 — signals, not labels).
    food: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )

    timestamp_s: float = 0.0


# ---------------------------------------------------------------------------
# ObsConfigV5 — every constant cited to its probe measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObsConfigV5:
    # Food delivery box measured ~±2100-2300 u but density is only flat
    # (~9.3e-5 /u^2) to r≈1250 before truncating -> window inside the
    # reliable zone.
    food_window: float = 1300.0
    # Density x window area ≈ 9.3e-5 * pi * 1300^2 ≈ 490 in window; keep the
    # nearest 128 (median in-window count over the session was ~609 for the
    # full ±2100 box).
    k_food: int = 128
    # Enemy heads delivered to median 2315 / p90 3338 / max 5647 u.
    enemy_window: float = 2500.0
    # Snakes visible: mean 12.4, max 23 over 1,947 snapshots -> 24 covers max.
    k_enemies: int = 24
    # Collision radar window (plan: kept from V4 in spirit, radius 800 u).
    danger_window: float = 800.0
    # Nearest enemy body points kept in the danger channel.
    k_danger: int = 96
    # Own body samples (V4 value, unchanged in spirit).
    k_own_body: int = 32
    # Client delivers enemy bodies decimated to median 64.5 u spacing.
    body_sample_spacing: float = 64.0
    # Body samples per enemy row, at fixed 64 u arc spacing from the head
    # (matches the client's delivered decimation).
    enemy_body_samples: int = 12
    # Client delivers at most ~96 pts per enemy snake.
    enemy_body_max_pts: int = 96
    # Food value measured min 3.0 / median 5.2 / max 14.2 -> /15 stays <1.
    value_norm: float = 15.0
    # Measured real map radius (border hit at r=14976.5). Used by the BRIDGE
    # to fill VisibleState.map_radius; build_obs normalizes r by
    # VisibleState.map_radius so the boundary scalar is scale-free in both
    # worlds (world scale is a knob, not a constant).
    map_radius_norm: float = 15000.0
    # Measured base speed 181.5 u/s (8-game value) -> sp/sp_base reads 1.0 at
    # base and ~2.055 while boosting.
    sp_base: float = 181.5
    # Mass log reference: log(mass / mass_norm), V4 style (initial mass 10;
    # real_mass(sct=2, fam=0) = 10.08, so spawn reads ~0).
    mass_norm: float = 10.0
    # Divisor for body radii (danger_segments[:,2] and the enemy radius
    # feature). V4's danger_radius_norm, kept: real_sc max reachable half-width
    # ~6.5 * 3.45 = 22.4 u, so /20 stays O(1).
    danger_radius_norm: float = 20.0
    # real_sc body-width law (measured EXACT, zero residual, 94 sizes,
    # sct 2-262): half-width = body_radius_base * (1 + (sct-2)/106).
    # body_radius_base is the R0 anchor from core/realism.py (NOT measured;
    # pinned by the scale-free spawn-width/turn-radius invariant).
    body_radius_base: float = 6.5
    body_radius_sct_offset: float = 2.0
    body_radius_sct_divisor: float = 106.0
    # Client mass-LUT generator params (bitwise-verified vs report.config).
    mscps: int = CLIENT_MSCPS
    lut_len: int = CLIENT_LUT_LEN

    @property
    def enemy_features(self) -> int:
        # head rel(2) + cos/sin(2) + sp(1) + mass log(1) + radius(1)
        # + body samples(2 * enemy_body_samples) + is_active(1)
        return 8 + 2 * self.enemy_body_samples

    @property
    def self_state_size(self) -> int:
        # r, cos/sin bearing-to-center, cos/sin ang, mass log, sp, boosting,
        # sc, last_action(3)
        return 12


# ---------------------------------------------------------------------------
# Builder helpers (pure)
# ---------------------------------------------------------------------------


def _real_sc_radius(sct: float, cfg: ObsConfigV5) -> float:
    """Body half-width under the measured real_sc law (world units)."""
    sc = 1.0 + (sct - cfg.body_radius_sct_offset) / cfg.body_radius_sct_divisor
    return cfg.body_radius_base * sc


def _arc_resample(
    pts: NDArray[np.float32], distances: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Sample a head-first polyline at given arc distances from the head.

    Returns ((len(distances), 2) points, valid mask). Distances beyond the
    polyline's total arc length are invalid (caller zero-pads).
    """
    n = len(pts)
    out = np.zeros((len(distances), 2), dtype=np.float64)
    if n == 0:
        return out, np.zeros(len(distances), dtype=np.bool_)
    p = pts.astype(np.float64)
    seg = np.diff(p, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    valid = distances <= cum[-1]
    out[:, 0] = np.interp(distances, cum, p[:, 0])
    out[:, 1] = np.interp(distances, cum, p[:, 1])
    return out, valid


def _nearest_first(
    rel: NDArray[np.float64], window: float, k: int
) -> NDArray[np.intp]:
    """Indices of the k nearest rows of `rel` within `window`, stably sorted
    nearest-first (ties broken by input order — order-canonical, P2)."""
    dists = np.hypot(rel[:, 0], rel[:, 1])
    within = np.flatnonzero(dists < window)
    order = within[np.argsort(dists[within], kind="stable")]
    return order[:k]


# ---------------------------------------------------------------------------
# build_obs — the one canonical builder
# ---------------------------------------------------------------------------


def build_obs(
    vs: VisibleState, cfg: ObsConfigV5 = ObsConfigV5()
) -> dict[str, NDArray[np.float32]]:
    """Pure function VisibleState -> observation dict. No state, no RNG.

    Returns {"self_state": (12,), "food": (k_food, 3),
             "enemies": (k_enemies, enemy_features),
             "danger_segments": (k_danger, 3), "own_body": (k_own_body, 2)}.

    self_state[9:12] is a last-action placeholder (heading cos/sin + boost)
    left at 0 — the CALLER (env / bridge loop) fills it, because the last
    commanded action is loop state, not visible state.
    """
    head = np.array([vs.head_x, vs.head_y], dtype=np.float64)

    # 1. self_state (12)
    dx = vs.head_x - vs.map_center_x
    dy = vs.head_y - vs.map_center_y
    r = math.hypot(dx, dy)
    if r > 0.0:
        cos_to_center, sin_to_center = -dx / r, -dy / r
    else:
        cos_to_center, sin_to_center = 0.0, 0.0
    mass = real_mass(vs.sct, vs.fam)
    self_state = np.array([
        r / vs.map_radius,
        cos_to_center,
        sin_to_center,
        math.cos(vs.angle),
        math.sin(vs.angle),
        math.log(max(mass, 1.0) / cfg.mass_norm),
        vs.speed / cfg.sp_base,
        1.0 if vs.boosting else 0.0,
        vs.scale,
        0.0,  # last_action heading cos — caller fills
        0.0,  # last_action heading sin — caller fills
        0.0,  # last_action boost — caller fills
    ], dtype=np.float32)

    # 2. food (k_food, 3) — nearest-first within food_window
    food_obs = np.zeros((cfg.k_food, 3), dtype=np.float32)
    if len(vs.food) > 0:
        rel = vs.food[:, :2].astype(np.float64) - head
        take = _nearest_first(rel, cfg.food_window, cfg.k_food)
        n = len(take)
        food_obs[:n, 0] = rel[take, 0] / cfg.food_window
        food_obs[:n, 1] = rel[take, 1] / cfg.food_window
        food_obs[:n, 2] = vs.food[take, 2] / cfg.value_norm

    # 3. enemies (k_enemies, F) — nearest-first by head distance
    enemy_obs = np.zeros((cfg.k_enemies, cfg.enemy_features), dtype=np.float32)
    if vs.enemies:
        heads = np.array(
            [[e.head_x, e.head_y] for e in vs.enemies], dtype=np.float64
        )
        take = _nearest_first(heads - head, cfg.enemy_window, cfg.k_enemies)
        sample_d = cfg.body_sample_spacing * np.arange(
            1, cfg.enemy_body_samples + 1, dtype=np.float64
        )
        for slot, idx in enumerate(take):
            e = vs.enemies[idx]
            row = enemy_obs[slot]
            row[0] = (e.head_x - vs.head_x) / cfg.enemy_window
            row[1] = (e.head_y - vs.head_y) / cfg.enemy_window
            row[2] = math.cos(e.angle)
            row[3] = math.sin(e.angle)
            row[4] = e.speed / cfg.sp_base
            # No enemy fam from the client -> mass from sct alone.
            e_mass = real_mass(e.sct, 0.0)
            row[5] = math.log(max(e_mass, 1.0) / cfg.mass_norm)
            row[6] = _real_sc_radius(e.sct, cfg) / cfg.danger_radius_norm
            # 12 body samples at fixed arc spacing from the enemy's head,
            # relative to OUR head (head itself is already features 0-1).
            samples, valid = _arc_resample(e.pts, sample_d)
            for j in range(cfg.enemy_body_samples):
                if valid[j]:
                    row[7 + 2 * j] = (samples[j, 0] - vs.head_x) / cfg.enemy_window
                    row[8 + 2 * j] = (samples[j, 1] - vs.head_y) / cfg.enemy_window
            row[7 + 2 * cfg.enemy_body_samples] = 1.0  # is_active

    # 4. danger_segments (k_danger, 3) — nearest enemy body points
    danger_obs = np.zeros((cfg.k_danger, 3), dtype=np.float32)
    if vs.enemies:
        pts_list = [e.pts for e in vs.enemies if len(e.pts) > 0]
        if pts_list:
            all_pts = np.concatenate(pts_list, axis=0).astype(np.float64)
            radii = np.concatenate([
                np.full(
                    len(e.pts),
                    _real_sc_radius(e.sct, cfg),
                    dtype=np.float64,
                )
                for e in vs.enemies
                if len(e.pts) > 0
            ])
            rel = all_pts - head
            take = _nearest_first(rel, cfg.danger_window, cfg.k_danger)
            n = len(take)
            danger_obs[:n, 0] = rel[take, 0] / cfg.danger_window
            danger_obs[:n, 1] = rel[take, 1] / cfg.danger_window
            danger_obs[:n, 2] = radii[take] / cfg.danger_radius_norm

    # 5. own_body (k_own_body, 2) — fixed-arc-spacing samples of own pts,
    # from the head (head excluded — it is self_state), rel / food_window
    # (the plan's own-body norm).
    own_obs = np.zeros((cfg.k_own_body, 2), dtype=np.float32)
    if len(vs.body_pts) > 0:
        sample_d = cfg.body_sample_spacing * np.arange(
            1, cfg.k_own_body + 1, dtype=np.float64
        )
        samples, valid = _arc_resample(vs.body_pts, sample_d)
        rel = samples - head
        own_obs[valid, 0] = (rel[valid, 0] / cfg.food_window).astype(np.float32)
        own_obs[valid, 1] = (rel[valid, 1] / cfg.food_window).astype(np.float32)

    return {
        "self_state": self_state,
        "food": food_obs,
        "enemies": enemy_obs,
        "danger_segments": danger_obs,
        "own_body": own_obs,
    }
