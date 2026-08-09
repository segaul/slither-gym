import json
import math
import os

import numpy as np
import pytest

from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World
from slither_gym.obs.schema_v5 import (
    EnemyVisible,
    ObsConfigV5,
    VisibleState,
    build_mass_luts,
    build_obs,
    real_mass,
)
from slither_gym.obs.visibility import visible_state_from_world

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "probe_v2_snapshot.json")
FULL_CAPTURE = os.path.expanduser(
    "~/Downloads/slither_probe_v2_2026-08-09T03-22-10_game01.json"
)

# Client sp -> u/s: measured base sp 5.78 corresponds to 181.5 u/s.
_SP_TO_UPS = 181.5 / 5.78


def _simple_state(**overrides: object) -> VisibleState:
    """A small hand-built VisibleState. All coords on a 0.25 grid so a +1000
    shift is exact in float32 (translation-invariance test)."""
    fields: dict[str, object] = dict(
        head_x=1000.0,
        head_y=2000.0,
        angle=0.5,
        speed=181.5,
        sct=10,
        fam=0.25,
        scale=1.0754716981132075,
        boosting=False,
        body_pts=np.array(
            [[1000.0, 2000.0], [960.0, 2000.0], [920.0, 2000.0],
             [880.0, 2000.0], [840.0, 2000.0]],
            dtype=np.float32,
        ),
        map_center_x=0.0,
        map_center_y=0.0,
        map_radius=15000.0,
        enemies=(
            EnemyVisible(
                snake_id=7,
                head_x=1400.0,
                head_y=2000.0,
                angle=1.0,
                speed=181.5,
                sct=50,
                pts=np.array(
                    [[1400.0, 2000.0], [1464.0, 2000.0], [1528.0, 2000.0],
                     [1592.0, 2000.0]],
                    dtype=np.float32,
                ),
            ),
            EnemyVisible(
                snake_id=3,
                head_x=1200.0,
                head_y=2000.0,
                angle=2.0,
                speed=373.0,
                sct=20,
                pts=np.array(
                    [[1200.0, 2000.0], [1200.0, 2064.0]], dtype=np.float32,
                ),
            ),
        ),
        food=np.array(
            [[1100.0, 2000.0, 5.0], [1050.0, 2000.0, 3.0], [1500.0, 2500.0, 14.0]],
            dtype=np.float32,
        ),
        timestamp_s=1.0,
    )
    fields.update(overrides)
    return VisibleState(**fields)  # type: ignore[arg-type]


def _shift_state(vs: VisibleState, dx: float, dy: float, shift_center: bool) -> VisibleState:
    d = np.array([dx, dy], dtype=np.float32)
    enemies = tuple(
        EnemyVisible(
            snake_id=e.snake_id,
            head_x=e.head_x + dx,
            head_y=e.head_y + dy,
            angle=e.angle,
            speed=e.speed,
            sct=e.sct,
            pts=e.pts + d,
        )
        for e in vs.enemies
    )
    food = vs.food.copy()
    if len(food):
        food[:, 0] += dx
        food[:, 1] += dy
    return VisibleState(
        head_x=vs.head_x + dx,
        head_y=vs.head_y + dy,
        angle=vs.angle,
        speed=vs.speed,
        sct=vs.sct,
        fam=vs.fam,
        scale=vs.scale,
        boosting=vs.boosting,
        body_pts=vs.body_pts + d,
        map_center_x=vs.map_center_x + (dx if shift_center else 0.0),
        map_center_y=vs.map_center_y + (dy if shift_center else 0.0),
        map_radius=vs.map_radius,
        enemies=enemies,
        food=food,
        timestamp_s=vs.timestamp_s,
    )


# ---------------------------------------------------------------------------
# Shapes / dtypes / finiteness
# ---------------------------------------------------------------------------


def test_shapes_dtypes_finite() -> None:
    cfg = ObsConfigV5()
    obs = build_obs(_simple_state(), cfg)
    assert set(obs) == {"self_state", "food", "enemies", "danger_segments", "own_body"}
    assert obs["self_state"].shape == (12,)
    assert obs["food"].shape == (cfg.k_food, 3)
    assert obs["enemies"].shape == (cfg.k_enemies, cfg.enemy_features)
    assert obs["enemies"].shape == (24, 32)
    assert obs["danger_segments"].shape == (cfg.k_danger, 3)
    assert obs["own_body"].shape == (cfg.k_own_body, 2)
    for name, arr in obs.items():
        assert arr.dtype == np.float32, name
        assert np.all(np.isfinite(arr)), name


def test_self_state_contents() -> None:
    vs = _simple_state(boosting=True)
    obs = build_obs(vs)
    s = obs["self_state"]
    r = math.hypot(1000.0, 2000.0)
    np.testing.assert_allclose(s[0], r / 15000.0, rtol=1e-6)
    # bearing-to-center points back at the origin
    np.testing.assert_allclose(s[1], -1000.0 / r, rtol=1e-6)
    np.testing.assert_allclose(s[2], -2000.0 / r, rtol=1e-6)
    np.testing.assert_allclose(s[3], math.cos(0.5), rtol=1e-6)
    np.testing.assert_allclose(s[4], math.sin(0.5), rtol=1e-6)
    np.testing.assert_allclose(
        s[5], math.log(real_mass(10, 0.25) / 10.0), rtol=1e-6
    )
    np.testing.assert_allclose(s[6], 1.0, rtol=1e-6)  # 181.5 / sp_base
    assert s[7] == 1.0  # boosting
    np.testing.assert_allclose(s[8], vs.scale, rtol=1e-6)
    # last-action placeholder left for the caller
    assert s[9] == s[10] == s[11] == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_identical_bytes() -> None:
    vs = _simple_state()
    a = build_obs(vs)
    b = build_obs(vs)
    for name in a:
        assert a[name].tobytes() == b[name].tobytes(), name


# ---------------------------------------------------------------------------
# Translation invariance
# ---------------------------------------------------------------------------


def test_translation_invariance() -> None:
    vs = _simple_state()
    a = build_obs(vs)
    # Shift entities but NOT map center: only the self_state position scalars
    # (r, bearing-to-center) may change.
    b = build_obs(_shift_state(vs, 1000.0, 1000.0, shift_center=False))
    for name in ("food", "enemies", "danger_segments", "own_body"):
        assert a[name].tobytes() == b[name].tobytes(), name
    assert not np.array_equal(a["self_state"][:3], b["self_state"][:3])
    assert a["self_state"][3:].tobytes() == b["self_state"][3:].tobytes()
    # Shift the whole world INCLUDING the center: fully identical.
    c = build_obs(_shift_state(vs, 1000.0, 1000.0, shift_center=True))
    for name in a:
        assert a[name].tobytes() == c[name].tobytes(), name


# ---------------------------------------------------------------------------
# Sorting / windows
# ---------------------------------------------------------------------------


def test_food_nearest_first_and_window() -> None:
    obs = build_obs(_simple_state())
    food = obs["food"]
    # nearest food is at +50 u in x (1050) with value 3, then +100 u (1100).
    np.testing.assert_allclose(food[0, 0], 50.0 / 1300.0, rtol=1e-5)
    np.testing.assert_allclose(food[0, 2], 3.0 / 15.0, rtol=1e-5)
    np.testing.assert_allclose(food[1, 0], 100.0 / 1300.0, rtol=1e-5)
    np.testing.assert_allclose(food[1, 2], 5.0 / 15.0, rtol=1e-5)
    # third pellet is at dist hypot(500, 500) = 707 -> inside the window
    assert food[2, 2] > 0.0
    assert np.all(food[3:] == 0.0)


def test_food_outside_window_dropped() -> None:
    vs = _simple_state(
        food=np.array([[5000.0, 2000.0, 5.0]], dtype=np.float32)
    )
    obs = build_obs(vs)
    assert np.all(obs["food"] == 0.0)


def test_enemies_nearest_first_by_head() -> None:
    obs = build_obs(_simple_state())
    enemies = obs["enemies"]
    # snake_id 3 (head at +200 u) must occupy slot 0, snake 7 (+400 u) slot 1.
    np.testing.assert_allclose(enemies[0, 0], 200.0 / 2500.0, rtol=1e-5)
    np.testing.assert_allclose(enemies[1, 0], 400.0 / 2500.0, rtol=1e-5)
    # is_active on both, zero elsewhere
    assert enemies[0, 31] == 1.0 and enemies[1, 31] == 1.0
    assert np.all(enemies[2:] == 0.0)
    # speed carries the boost signal implicitly: slot 0 boosts (373 u/s)
    np.testing.assert_allclose(enemies[0, 4], 373.0 / 181.5, rtol=1e-5)
    # mass log from sct alone via the real LUTs
    np.testing.assert_allclose(
        enemies[0, 5], math.log(real_mass(20, 0.0) / 10.0), rtol=1e-5
    )
    # body radius via real_sc law / 20
    np.testing.assert_allclose(
        enemies[0, 6], 6.5 * (1.0 + (20 - 2) / 106.0) / 20.0, rtol=1e-5
    )


def test_enemy_outside_window_dropped() -> None:
    far = EnemyVisible(
        snake_id=9, head_x=9000.0, head_y=2000.0, angle=0.0, speed=181.5,
        sct=10, pts=np.array([[9000.0, 2000.0]], dtype=np.float32),
    )
    obs = build_obs(_simple_state(enemies=(far,)))
    assert np.all(obs["enemies"] == 0.0)
    assert np.all(obs["danger_segments"] == 0.0)


def test_enemy_body_samples_fixed_arc_spacing() -> None:
    obs = build_obs(_simple_state())
    row = obs["enemies"][1]  # snake 7: straight body along +x, 64 u spacing
    # sample j sits at arc 64*(j+1) from the enemy head (1400, 2000)
    for j in range(3):  # its body is 192 u long -> 3 valid samples
        np.testing.assert_allclose(
            row[7 + 2 * j], (400.0 + 64.0 * (j + 1)) / 2500.0, rtol=1e-5,
        )
        np.testing.assert_allclose(row[8 + 2 * j], 0.0, atol=1e-6)
    # beyond the body: zero-padded
    assert np.all(row[7 + 2 * 3 : 7 + 2 * 12] == 0.0)


def test_danger_segments_nearest_first() -> None:
    obs = build_obs(_simple_state())
    danger = obs["danger_segments"]
    # nearest enemy body point: snake 3's head at +200 u
    np.testing.assert_allclose(danger[0, 0], 200.0 / 800.0, rtol=1e-5)
    np.testing.assert_allclose(
        danger[0, 2], 6.5 * (1.0 + (20 - 2) / 106.0) / 20.0, rtol=1e-5,
    )
    dists = np.hypot(
        danger[:, 0].astype(np.float64), danger[:, 1].astype(np.float64)
    )
    n_real = int(np.sum(np.any(danger != 0.0, axis=1)))
    assert n_real == 6  # 2 + 4 delivered points, all within 800 u
    assert np.all(np.diff(dists[:n_real]) >= -1e-7)  # sorted nearest-first
    assert np.all(danger[n_real:] == 0.0)


def test_own_body_arc_samples() -> None:
    obs = build_obs(_simple_state())
    own = obs["own_body"]
    # body runs -x at 40 u/segment, 160 u total -> samples at 64 and 128 u
    np.testing.assert_allclose(own[0, 0], -64.0 / 1300.0, rtol=1e-5)
    np.testing.assert_allclose(own[1, 0], -128.0 / 1300.0, rtol=1e-5)
    assert np.all(own[2:] == 0.0)


def test_truncation_zero_padded_and_capped() -> None:
    rng = np.random.default_rng(0)
    n = 500
    food = np.zeros((n, 3), dtype=np.float32)
    ang = rng.uniform(0, 2 * math.pi, n)
    r = rng.uniform(10, 1200, n)
    food[:, 0] = 1000.0 + r * np.cos(ang)
    food[:, 1] = 2000.0 + r * np.sin(ang)
    food[:, 2] = rng.uniform(3, 14, n)
    obs = build_obs(_simple_state(food=food))
    assert obs["food"].shape == (128, 3)
    # every slot filled, nearest-first
    assert np.all(np.any(obs["food"] != 0.0, axis=1))
    dists = np.hypot(obs["food"][:, 0], obs["food"][:, 1])
    assert np.all(np.diff(dists) >= -1e-6)


def test_empty_state_all_zero_but_self() -> None:
    vs = _simple_state(
        enemies=(),
        food=np.zeros((0, 3), dtype=np.float32),
        body_pts=np.zeros((0, 2), dtype=np.float32),
    )
    obs = build_obs(vs)
    for name in ("food", "enemies", "danger_segments", "own_body"):
        assert np.all(obs[name] == 0.0), name
    assert np.all(np.isfinite(obs["self_state"]))


# ---------------------------------------------------------------------------
# real_mass — hand-checked against the W32 fam-wrap captains-log entry
# ---------------------------------------------------------------------------


def test_real_mass_fam_wrap_continuity() -> None:
    # docs/captains-log/2026-W32.md: sct 4 -> 5 lines up exactly with fam
    # 0.9619 -> 0.1171, i.e. total 4.9619 -> 5.1171, a normal +0.155 gain.
    m1 = real_mass(4, 0.9619)
    m2 = real_mass(5, 0.1171)
    assert m2 > m1
    # ~0.155 total-units * 15 mass-units each ~= 2.3; the wrap must NOT look
    # like the +-0.8-1.0 spurious spike the raw fam diff produced.
    assert 1.5 < (m2 - m1) < 3.0
    # spot values (fpsls[4]=4.03167, fmlts[4]=0.97919, fpsls[5]=5.05293,
    # fmlts[5]=0.97403) -> hand-computed masses
    np.testing.assert_allclose(m1, 55.21, atol=0.05)
    np.testing.assert_allclose(m2, 57.60, atol=0.05)


def test_real_mass_spawn_is_initial_mass() -> None:
    # fpsls[2] = 2.00525 -> (2.00525 - 1)*15 - 5 = 10.08 ~= sim initial 10.
    np.testing.assert_allclose(real_mass(2, 0.0), 10.08, atol=0.01)


def test_mass_luts_match_committed_spot_checks() -> None:
    with open(FIXTURE) as f:
        fixture = json.load(f)
    cfg = fixture["config"]
    fpsls, fmlts = build_mass_luts(cfg["mscps"], cfg["lut_len"])
    for idx, vals in cfg["lut_spot_checks"].items():
        i = int(idx)
        assert fmlts[i] == vals["fmlts"], i
        assert fpsls[i] == vals["fpsls"], i


# ---------------------------------------------------------------------------
# Probe-capture smoke tests
# ---------------------------------------------------------------------------


def _visible_state_from_fixture(fixture: dict) -> VisibleState:
    cfg = fixture["config"]
    grd = float(cfg["grd"])
    me = fixture["me"]
    # Client pts arrive TAIL-first; VisibleState is canonical head-first.
    own_pts = np.array(me["pts"], dtype=np.float32)[::-1].copy()
    enemies = tuple(
        EnemyVisible(
            snake_id=int(s["id"]),
            head_x=float(s["xx"]),
            head_y=float(s["yy"]),
            angle=float(s["ang"]),
            speed=float(s["sp"]) * _SP_TO_UPS,
            sct=int(s["sct"]),
            pts=np.array(s["pts"], dtype=np.float32)[::-1].copy(),
        )
        for s in fixture["snakes"]
    )
    return VisibleState(
        head_x=float(me["xx"]),
        head_y=float(me["yy"]),
        angle=float(me["ang"]),
        speed=float(me["sp"]) * _SP_TO_UPS,
        sct=int(me["sct"]),
        fam=float(fixture["me_fam"]),
        scale=float(me["sc"]),
        boosting=bool(fixture["me_boosting"]),
        body_pts=own_pts,
        map_center_x=grd,
        map_center_y=grd,
        map_radius=15000.0,
        enemies=enemies,
        food=np.array(fixture["food"], dtype=np.float32),
        timestamp_s=float(fixture["snapshot_t"]),
    )


def test_fixture_snapshot_smoke() -> None:
    with open(FIXTURE) as f:
        fixture = json.load(f)
    vs = _visible_state_from_fixture(fixture)
    obs = build_obs(vs)
    s = obs["self_state"]
    assert np.all(np.isfinite(s))
    assert 0.0 <= s[0] <= 1.0  # inside the map
    # real snapshot has food and enemies in range
    assert np.any(obs["food"] != 0.0)
    assert np.any(obs["enemies"][:, -1] == 1.0)
    # normalized rel positions stay in the unit disc of their window
    for name, cols in (("food", 2), ("danger_segments", 2)):
        rel = obs[name][:, :cols]
        assert np.all(np.hypot(rel[:, 0], rel[:, 1]) <= 1.0 + 1e-6), name
    # food values on the real 3-14 scale -> /15 in (0, 1)
    vals = obs["food"][np.any(obs["food"] != 0.0, axis=1), 2]
    assert np.all((vals > 0.1) & (vals < 1.0))
    # deterministic on the real snapshot too
    obs2 = build_obs(vs)
    for name in obs:
        assert obs[name].tobytes() == obs2[name].tobytes(), name


@pytest.mark.skipif(
    not os.path.exists(FULL_CAPTURE), reason="full probe capture not present"
)
def test_full_capture_luts_and_snapshot() -> None:
    with open(FULL_CAPTURE) as f:
        capture = json.load(f)
    cfg = capture["report"]["config"]
    # The generated LUTs must reproduce the client's tables BITWISE.
    fpsls, fmlts = build_mass_luts(cfg["mscps"], len(cfg["fmlts"]))
    assert np.array_equal(fmlts, np.array(cfg["fmlts"], dtype=np.float64))
    assert np.array_equal(fpsls, np.array(cfg["fpsls"], dtype=np.float64))
    # Build obs from a real full-capture snapshot (no committed extraction).
    snap = max(capture["snapshots"], key=lambda s: len(s["snakes"]))
    sample = min(
        (x for x in capture["samples"] if x["life"] == snap["life"]),
        key=lambda x: abs(x["t"] - snap["t"]),
    )
    fixture = {
        "config": {"grd": cfg["grd"]},
        "me": snap["me"],
        "me_fam": sample["fam"],
        "me_boosting": sample["boosting"],
        "snakes": snap["snakes"],
        "food": snap["food"],
        "snapshot_t": snap["t"],
    }
    obs = build_obs(_visible_state_from_fixture(fixture))
    for arr in obs.values():
        assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# Sim visibility mask
# ---------------------------------------------------------------------------


def test_visible_state_from_world() -> None:
    config = WorldConfig()
    world = World(config, seed=3)
    world.spawn_snake(0)
    world.spawn_snake(1)
    for _ in range(20):
        world.step({0: (1.0, 0.0, False), 1: (0.0, 1.0, True)})

    cfg = ObsConfigV5()
    vs = visible_state_from_world(world, 0, cfg)
    me = world.get_snake_states()[0]

    assert vs.head_x == me.head_x and vs.head_y == me.head_y
    assert vs.sct == me.segment_count
    assert vs.boosting == me.boosting  # commanded action, not a heuristic
    assert vs.map_center_x == 0.0 and vs.map_center_y == 0.0
    assert vs.map_radius == config.map_radius
    np.testing.assert_allclose(vs.speed, me.speed * 40.0)  # u/tick -> u/s
    np.testing.assert_allclose(vs.timestamp_s, world.get_tick() / 40.0)

    # visibility masking honored
    head = np.array([me.head_x, me.head_y])
    if len(vs.food):
        d = np.hypot(vs.food[:, 0] - head[0], vs.food[:, 1] - head[1])
        assert np.all(d < cfg.food_window)
    for e in vs.enemies:
        assert math.hypot(e.head_x - head[0], e.head_y - head[1]) < cfg.enemy_window
        # decimated to the client's ~64 u spacing
        if len(e.pts) > 1:
            seg = np.diff(e.pts.astype(np.float64), axis=0)
            steps = np.hypot(seg[:, 0], seg[:, 1])
            assert np.all(steps <= cfg.body_sample_spacing + 1.0)
        assert len(e.pts) <= cfg.enemy_body_max_pts

    obs = build_obs(vs, cfg)
    for arr in obs.values():
        assert np.all(np.isfinite(arr))
    # the whole sim path is deterministic
    vs2 = visible_state_from_world(world, 0, cfg)
    obs2 = build_obs(vs2, cfg)
    for name in obs:
        assert obs[name].tobytes() == obs2[name].tobytes(), name


def test_visibility_masks_far_enemy() -> None:
    config = WorldConfig(map_radius=10000.0)
    world = World(config, seed=0)
    world.spawn_snake(0)
    # place the second snake far outside the enemy window
    world.spawn_snake(1)
    states = world.get_snake_states()
    d = math.hypot(
        states[1].head_x - states[0].head_x,
        states[1].head_y - states[0].head_y,
    )
    vs = visible_state_from_world(world, 0)
    if d >= 2500.0:
        assert vs.enemies == ()
    else:
        assert len(vs.enemies) == 1
