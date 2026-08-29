"""V5 observation package — deployment-first, one builder for sim and bridge.

See docs/STATE_SPACE_V5_PLAN.md (slither-rl repo). `schema_v5` holds the
canonical `VisibleState` -> obs builder; `visibility` masks the sim's
omniscient World down to the measured client delivery model.
"""

from slither_gym.obs.schema_v5 import (
    EnemyVisible,
    ObsConfigV5,
    VisibleState,
    build_mass_luts,
    build_obs,
    real_mass,
)
from slither_gym.obs.visibility import visible_state_from_world

__all__ = [
    "EnemyVisible",
    "ObsConfigV5",
    "VisibleState",
    "build_mass_luts",
    "build_obs",
    "real_mass",
    "visible_state_from_world",
]
