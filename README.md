# slither-gym

Gymnasium/PettingZoo-compatible Slither.io training environment for reinforcement learning.

## Quick Start

```bash
uv sync
uv run python demo.py          # visual demo (Pygame)
uv run pytest tests/ -v         # run tests
uv run mypy src/slither_gym/ --strict  # type check
```

## Demo Controls

| Key     | Action          |
|---------|-----------------|
| Mouse   | Steer snake     |
| Space   | Boost (costs mass) |
| R       | Reset           |
| Esc / Q | Quit            |

## Usage

### Single-agent (Gymnasium)

```python
from slither_gym.rl.env_gym import SlitherGymEnv

env = SlitherGymEnv(num_bots=10, seed=42)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # [cos, sin, boost]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Multi-agent (PettingZoo)

```python
from slither_gym.rl.env_parallel import SlitherParallelEnv

env = SlitherParallelEnv(num_agents=4, seed=42)
obs, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
```

## Observation Space

| Key          | Shape          | Description |
|-------------|----------------|-------------|
| `self_state` | `(6,)`         | norm_x, norm_y, cos(angle), sin(angle), log(mass), speed |
| `food`       | `(32, 3)`      | K-nearest food: rel_x, rel_y, value |
| `enemies`    | `(64, 7)`      | K-nearest enemy segments: rel_x, rel_y, is_head, log(mass), speed, rel_velocity_angle, radius |
| `minimap`    | `(32, 32)`     | Circular grid of snake mass density across the full map |

## Action Space

Continuous `Box(3,)`: `[cos, sin, boost]`
- `cos, sin` — target direction (normalized internally)
- `boost` — `> 0.5` activates boost (2x speed, costs mass)

## Package Structure

```
slither_gym/
├── demo.py             # Run the game simulation and control it yourself
├── core/               # Pure game simulation (no RL dependencies)
│   ├── types.py        # WorldConfig, SnakeState, StepResult
│   ├── snake.py        # SnakeManager — movement, growth, death
│   ├── food.py         # FoodManager — spawn, collect, free-list
│   ├── spatial_hash.py # Grid-based spatial index for collision
│   ├── minimap.py      # Circular minimap density grid
│   └── world.py        # World — orchestrates one physics tick
└── rl/                 # RL interface layer
    ├── types.py        # ObsConfig, RawGameState, AgentId
    ├── obs_processor.py# Observation computation (pure function)
    ├── reward.py       # Reward computation (pure function)
    ├── bot_policy.py   # Rule-based bot (flee > seek food > wander)
    ├── env_parallel.py # PettingZoo ParallelEnv
    └── env_gym.py      # Gymnasium single-agent wrapper
```

## Game Mechanics

- **Movement**: head advances at `speed`, each body segment follows the one ahead (max distance = `segment_spacing`)
- **Boost**: 2x speed, consumes mass at 5 segments/sec, floor at initial mass
- **Collision**: head hits another snake's body = death. Self-collision ignored.
- **Boundary**: leaving the map circle = death
- **Death**: all segments become food pellets (value scales with snake mass)
- **Food**: random food spawns across the map. Corpse food is larger and more valuable.
- **Growth**: eating food adds mass, segment count increases naturally as the tail extends

## Configuration

```python
from slither_gym.core.types import WorldConfig
from slither_gym.rl.types import ObsConfig

config = WorldConfig(
    map_radius=3000.0,
    max_snakes=32,
    base_speed=3.0,
    boost_speed=6.0,
    segment_spacing=5.0,
    step_mul=4,          # physics ticks per env.step()
    perception_radius=500.0,
)

obs_config = ObsConfig(
    k_food=32,
    k_enemies=64,
    minimap_size=32,
)
```
