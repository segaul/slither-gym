"""T28 corpse-TTL knob: corpse pellets despawn after ttl; floor pellets never; knob off
(None) leaves the expiry path unexecuted (byte-identity held by the goldens suite)."""
import numpy as np

from slither_gym.core.food import FoodManager
from slither_gym.core.types import WorldConfig


def test_expiry_despawns_corpses_only():
    fm = FoodManager(WorldConfig(max_food=64), np.random.default_rng(0))
    fm.now_tick = 0
    fm.spawn_at(10.0, 10.0, 5.0, corpse=True)
    fm.spawn_at(20.0, 20.0, 5.0, corpse=False)
    fm.now_tick = 50
    fm.spawn_at(30.0, 30.0, 5.0, corpse=True)   # young corpse
    assert fm.expire_corpses(40) == 1            # only the tick-0 corpse
    assert fm.alive_count() == 2
    assert fm.alive_floor_count() == 1
    fm.now_tick = 200
    assert fm.expire_corpses(40) == 1
    assert fm.alive_count() == 1                 # floor pellet survives forever


def test_env_knob_reaches_world():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../training"))
    from slither_train.env_factory import ENV_CONFIG_KEYS
    assert "corpse_ttl_ticks" in ENV_CONFIG_KEYS
