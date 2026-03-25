#!/usr/bin/env python3
"""
Visual e2e demo of slither-gym using Pygame.

Run: cd /Users/evansegaul/workspace/slither-rl/slither-gym && uv run python demo.py

Controls:
  Mouse  - steer snake toward cursor
  SPACE  - hold to boost
  ESC/Q  - quit
  R      - reset
"""
import math
import sys

import numpy as np
import pygame

from slither_gym.core.types import WorldConfig
from slither_gym.core.world import World
from slither_gym.rl.env_gym import SlitherGymEnv

SCREEN_W, SCREEN_H = 1100, 900
FPS = 40
NUM_BOTS = 25
ZOOM = 0.6

COLORS = [
    (80, 220, 80), (220, 80, 80), (80, 80, 220), (220, 180, 40),
    (180, 40, 220), (40, 220, 220), (220, 120, 80), (120, 220, 180),
    (200, 200, 120), (160, 80, 160), (255, 100, 150), (100, 255, 100),
    (100, 150, 255), (255, 200, 100), (200, 100, 255), (100, 255, 255),
]

BOT_MASS_WEIGHTS = [10] * 10 + [50] * 5 + [150] * 4 + [400] * 3 + [1000] * 2 + [3000] * 1


# --- Coordinate helpers ---

def world_to_screen(wx: float, wy: float, cam_x: float, cam_y: float) -> tuple[int, int]:
    return int(SCREEN_W / 2 + (wx - cam_x) * ZOOM), int(SCREEN_H / 2 + (wy - cam_y) * ZOOM)


def screen_to_world(sx: int, sy: int, cam_x: float, cam_y: float) -> tuple[float, float]:
    return cam_x + (sx - SCREEN_W / 2) / ZOOM, cam_y + (sy - SCREEN_H / 2) / ZOOM


# --- Input ---

def get_player_action(cam_x: float, cam_y: float, world: World) -> np.ndarray:
    keys = pygame.key.get_pressed()
    boost = keys[pygame.K_SPACE]

    player = world.get_snake_states().get(0)
    if player and player.alive:
        mx, my = pygame.mouse.get_pos()
        tx, ty = screen_to_world(mx, my, cam_x, cam_y)
        dx, dy = tx - player.head_x, ty - player.head_y
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 0:
            cos_a, sin_a = dx / mag, dy / mag
        else:
            cos_a, sin_a = math.cos(player.angle), math.sin(player.angle)
    else:
        cos_a, sin_a = 1.0, 0.0

    return np.array([cos_a, sin_a, 1.0 if boost else 0.0], dtype=np.float32)


# --- Rendering ---

def draw_boundary(screen: pygame.Surface, cam_x: float, cam_y: float, radius: float) -> None:
    bx, by = world_to_screen(0, 0, cam_x, cam_y)
    pygame.draw.circle(screen, (60, 20, 20), (bx, by), int(radius * ZOOM), 2)


def draw_food(screen: pygame.Surface, world: World, cam_x: float, cam_y: float) -> None:
    food_pos = world.get_food_positions()
    food_vals = world.get_food_values()
    for k in range(len(food_pos)):
        fx, fy = world_to_screen(float(food_pos[k, 0]), float(food_pos[k, 1]), cam_x, cam_y)
        if not (0 <= fx < SCREEN_W and 0 <= fy < SCREEN_H):
            continue
        val = float(food_vals[k])
        if val > 5.0:
            color, r = (255, 100, 40), max(4, int(5 * ZOOM))
        elif val > 3.0:
            color, r = (255, 160, 60), max(3, int(4 * ZOOM))
        elif val > 1.5:
            color, r = (240, 220, 80), max(2, int(3 * ZOOM))
        else:
            color, r = (200, 200, 60), max(2, int(2 * ZOOM))
        pygame.draw.circle(screen, color, (fx, fy), r)


def draw_snakes(screen: pygame.Surface, world: World, cam_x: float, cam_y: float) -> None:
    for sid, state in world.get_snake_states().items():
        if not state.alive:
            continue
        color = COLORS[sid % len(COLORS)]
        segs = world.get_segments(sid)
        r = max(2, int(state.segment_radius * ZOOM))

        for j in range(len(segs) - 1, -1, -1):
            sx, sy = world_to_screen(float(segs[j, 0]), float(segs[j, 1]), cam_x, cam_y)
            if -r <= sx < SCREEN_W + r and -r <= sy < SCREEN_H + r:
                pygame.draw.circle(screen, color, (sx, sy), r)

        hx, hy = world_to_screen(state.head_x, state.head_y, cam_x, cam_y)
        pygame.draw.circle(screen, (255, 255, 255), (hx, hy), r + 1, 1)


def draw_death_markers(
    screen: pygame.Surface,
    markers: list[tuple[float, float, int]],
    cam_x: float,
    cam_y: float,
    current_tick: int,
) -> None:
    """Draw pulsing X markers where snakes died recently."""
    for wx, wy, death_tick in markers:
        age = current_tick - death_tick
        if age > 120:  # fade after 3 seconds
            continue
        alpha = max(0, 255 - age * 2)
        sx, sy = world_to_screen(wx, wy, cam_x, cam_y)
        if not (0 <= sx < SCREEN_W and 0 <= sy < SCREEN_H):
            continue
        size = 8 + (age % 10)
        color = (255, 50, 50)
        pygame.draw.line(screen, color, (sx - size, sy - size), (sx + size, sy + size), 2)
        pygame.draw.line(screen, color, (sx - size, sy + size), (sx + size, sy - size), 2)


def draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    world: World,
    death_log: list[str],
) -> None:
    player = world.get_snake_states().get(0)
    if not player:
        return
    alive_count = sum(1 for s in world.get_snake_states().values() if s.alive)
    hud = f"mass: {player.mass:.0f}  segs: {player.segment_count}  tick: {world.get_tick()}  alive: {alive_count}/{1 + NUM_BOTS}  food: {world._food._count}"
    if player.boosting:
        hud += "  BOOST"
    screen.blit(font.render(hud, True, (200, 200, 200)), (10, 10))

    # Show recent deaths
    for i, msg in enumerate(death_log[-3:]):
        screen.blit(font.render(msg, True, (255, 150, 150)), (10, 30 + i * 18))


# --- Main loop ---

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("slither-gym — visual e2e demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    config = WorldConfig(max_snakes=32, step_mul=1)
    rng = np.random.default_rng(42)

    env = SlitherGymEnv(
        world_config=config,
        num_bots=NUM_BOTS,
        max_ticks=999999,
        seed=42,
        respawn_bots=True,
    )

    death_markers: list[tuple[float, float, int]] = []  # (x, y, tick)
    death_log: list[str] = []

    def reset_env() -> None:
        env.reset()
        assert env._world is not None
        for i in range(1, 1 + NUM_BOTS):
            mass = float(rng.choice(BOT_MASS_WEIGHTS))
            env._world.spawn_snake(i, mass=mass)
        death_markers.clear()
        death_log.clear()

    reset_env()
    cam_x, cam_y = 0.0, 0.0
    prev_alive: set[int] = set()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    reset_env()

        world = env._world
        assert world is not None

        # Track who's alive before step
        curr_alive = {s.snake_id for s in world.get_snake_states().values() if s.alive}
        # Save positions of all alive snakes before step (for death markers)
        pre_positions = {
            s.snake_id: (s.head_x, s.head_y, s.segment_count)
            for s in world.get_snake_states().values() if s.alive
        }

        action = get_player_action(cam_x, cam_y, world)
        _, _, terminated, truncated, _ = env.step(action)

        # Detect deaths
        post_alive = {s.snake_id for s in world.get_snake_states().values() if s.alive}
        died = curr_alive - post_alive
        tick = world.get_tick()
        for sid in died:
            if sid in pre_positions:
                hx, hy, segs = pre_positions[sid]
                death_markers.append((hx, hy, tick))
                death_log.append(f"[{tick}] snake_{sid} died ({segs} segs) at ({hx:.0f},{hy:.0f}) food={world._food._count}")

        if terminated or truncated:
            reset_env()

        world = env._world
        assert world is not None
        player = world.get_snake_states().get(0)
        if player and player.alive:
            cam_x, cam_y = player.head_x, player.head_y

        # Prune old death markers
        death_markers[:] = [(x, y, t) for x, y, t in death_markers if tick - t < 120]

        screen.fill((15, 15, 25))
        draw_boundary(screen, cam_x, cam_y, config.map_radius)
        draw_food(screen, world, cam_x, cam_y)
        draw_snakes(screen, world, cam_x, cam_y)
        draw_death_markers(screen, death_markers, cam_x, cam_y, tick)
        draw_hud(screen, font, world, death_log)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
