"""Mask the sim's omniscient World down to the measured client delivery model.

P3 (docs/STATE_SPACE_V5_PLAN.md): visibility is a measured quantity. The sim
emits a `VisibleState` containing only what the real client would have
delivered — food within the food window, enemy snakes whose heads are within
the enemy window, enemy bodies decimated to ~64 u spacing — and the canonical
`build_obs()` never sees beyond it.

NOT wired into the training env yet (that is S5/S6); V4 code paths are
untouched and E-series runs stay reproducible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from slither_gym.core.realism import TICK_HZ
from slither_gym.core.world import World
from slither_gym.obs.schema_v5 import EnemyVisible, ObsConfigV5, VisibleState


def _decimate_pts(
    pts: NDArray[np.float32], spacing: float, max_pts: int
) -> NDArray[np.float32]:
    """Resample a head-first polyline at fixed arc spacing, head included.

    Models the client's delivered decimation (median spacing 64.5 u,
    <=~96 pts per snake). Output stays head-first.
    """
    n = len(pts)
    if n <= 1:
        return pts.astype(np.float32).reshape(n, 2)
    p = pts.astype(np.float64)
    seg = np.diff(p, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    total = float(cum[-1])
    n_out = min(int(total / spacing) + 1, max_pts)
    distances = spacing * np.arange(n_out, dtype=np.float64)
    out = np.empty((n_out, 2), dtype=np.float64)
    out[:, 0] = np.interp(distances, cum, p[:, 0])
    out[:, 1] = np.interp(distances, cum, p[:, 1])
    return out.astype(np.float32)


def visible_state_from_world(
    world: World,
    snake_id: int,
    cfg: ObsConfigV5 = ObsConfigV5(),
) -> VisibleState:
    """Build the client-equivalent VisibleState for one sim snake.

    Access patterns mirror env_parallel._get_observations: snake states via
    get_snake_states(), bodies via get_segments() (head-first), food via the
    get_food_* accessors. The sim world is centred at (0, 0).

    Notes on field provenance:
    - speed: sim stores u/tick; the client scale is u/s -> multiply by TICK_HZ.
    - boosting: the snake's commanded boost (SnakeState.boosting), matching
      "own boost = our own last commanded action" — never an sp heuristic.
    - fam: the sim has no fractional-mass counter; sct carries the size signal
      and fam is 0.0 until the B-block economics unit models it.
    - scale: the measured-exact real_sc law, sc = 1 + (sct-2)/106.
    - food values: passed through as-is. They are on the real 3-14 scale only
      once P0.3 (B-block food economics) lands; this module does not rescale.
    """
    states = world.get_snake_states()
    me = states[snake_id]
    head = np.array([me.head_x, me.head_y], dtype=np.float64)

    # Food within the measured delivery window (client global cap ignored —
    # the measured in-window population never exceeded it).
    food_pos = world.get_food_positions()
    food_vals = world.get_food_values()
    if len(food_pos) > 0:
        rel = food_pos.astype(np.float64) - head
        within = np.hypot(rel[:, 0], rel[:, 1]) < cfg.food_window
        food = np.concatenate(
            [food_pos[within], food_vals[within, None]], axis=1
        ).astype(np.float32)
    else:
        food = np.zeros((0, 3), dtype=np.float32)

    # Enemy snakes with head within the enemy window, bodies decimated to the
    # client's delivered ~64 u spacing.
    enemies: list[EnemyVisible] = []
    for other_id, other in states.items():
        if other_id == snake_id or not other.alive:
            continue
        if (
            np.hypot(other.head_x - me.head_x, other.head_y - me.head_y)
            >= cfg.enemy_window
        ):
            continue
        segs = world.get_segments(other_id)
        if len(segs) == 0:
            continue
        enemies.append(EnemyVisible(
            snake_id=other_id,
            head_x=other.head_x,
            head_y=other.head_y,
            angle=other.angle,
            speed=other.speed * TICK_HZ,
            sct=other.segment_count,
            pts=_decimate_pts(
                segs, cfg.body_sample_spacing, cfg.enemy_body_max_pts
            ),
        ))

    sc = 1.0 + (
        me.segment_count - cfg.body_radius_sct_offset
    ) / cfg.body_radius_sct_divisor

    return VisibleState(
        head_x=me.head_x,
        head_y=me.head_y,
        angle=me.angle,
        speed=me.speed * TICK_HZ,
        sct=me.segment_count,
        fam=0.0,
        scale=sc,
        boosting=me.boosting,
        body_pts=world.get_segments(snake_id).astype(np.float32),
        map_center_x=0.0,
        map_center_y=0.0,
        map_radius=world.get_config().map_radius,
        enemies=tuple(enemies),
        food=food,
        timestamp_s=world.get_tick() / TICK_HZ,
    )
