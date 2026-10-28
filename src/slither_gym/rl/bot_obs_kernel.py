"""INFRA-batched-hotspots (bot-obs half): the scripted-bot enemy pass as a numba kernel.

`_compute_bot_observation_fast`'s enemy section is a per-bot Python loop over ALL snake
states — O(bots x snakes) = ~250k dict iterations per tick at 499 bots, the dominant
share of the measured 15.5% bot-obs wall slice. This kernel runs the whole pass for every
scripted bot in one call over packed float64 state arrays.

Bitwise contract (mirrors the reference loop in rl/env_gym.py exactly):
- cull test `dx*dx + dy*dy > cull_sq` in float64, same operand order;
- sort by `math.hypot(dx, dy)` (float64), STABLE in the packed scan order — the reference
  uses Python's stable list.sort over dict-insertion order, so ties resolve identically
  (insertion sort shifts only on strictly-greater);
- every row value is computed in float64 and rounds once at the float32 store, exactly
  as numpy does when assigning a Python float into a float32 row.
Gated by tests/test_bot_obs_kernel.py (per-bot bitwise vs the reference builder) and the
physics goldens. Fallback: numba missing or SLITHER_BOT_OBS_KERNEL=python.
"""
import math
import os

BOT_OBS_KERNEL = "python"

if os.environ.get("SLITHER_BOT_OBS_KERNEL", "numba") != "python":
    try:
        from numba import njit

        @njit(cache=True, fastmath=False)
        def enemy_rows_numba(  # pragma: no cover - mirrored by tests
            bot_rows, hx, hy, ang, sp, mass, boost, segc, alive,
            k_enemies, cull_sq, perception_radius, speed_norm,
            idx_buf, dist_buf, out,
        ):
            n = hx.shape[0]
            for b in range(bot_rows.shape[0]):
                i = bot_rows[b]
                bx = hx[i]
                by = hy[i]
                nd = 0
                for j in range(n):
                    if not alive[j] or j == i or segc[j] == 0:
                        continue
                    dx = hx[j] - bx
                    dy = hy[j] - by
                    if dx * dx + dy * dy > cull_sq:
                        continue
                    d = math.hypot(dx, dy)
                    k = nd
                    while k > 0 and dist_buf[k - 1] > d:
                        dist_buf[k] = dist_buf[k - 1]
                        idx_buf[k] = idx_buf[k - 1]
                        k -= 1
                    dist_buf[k] = d
                    idx_buf[k] = j
                    nd += 1
                m = nd if nd < k_enemies else k_enemies
                for slot in range(m):
                    j = idx_buf[slot]
                    out[b, slot, 0] = (hx[j] - bx) / perception_radius
                    out[b, slot, 1] = (hy[j] - by) / perception_radius
                    mj = mass[j] if mass[j] > 1.0 else 1.0
                    out[b, slot, 26] = math.log(mj / 10.0)
                    out[b, slot, 27] = sp[j] / speed_norm
                    out[b, slot, 28] = math.cos(ang[j])
                    out[b, slot, 29] = math.sin(ang[j])
                    out[b, slot, 30] = 1.0 if boost[j] else 0.0
                    out[b, slot, 31] = 1.0

        BOT_OBS_KERNEL = "numba-enemy-rows-v1"
    except ImportError:  # numba genuinely absent -> per-bot reference loop
        pass
