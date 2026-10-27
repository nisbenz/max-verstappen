"""Faster drawing of CarRacing frames, pixel for pixel identical (training only).

~90% of a training env step is spent drawing the observation, not in Box2D:
`_render_road` transforms ~750 grass and road polygons one vertex at a time
in Python just to find the ~80 that are on screen, and `_render` reloads the
HUD font from disk on every frame.

`install` replaces that with:
  * the same vertex transform for all polygons at once in numpy, then the
    same pygame draw calls, with the same arguments, for the same visible
    polygons, in the same order;
  * one font object, created once.
Physics is untouched. `scripts/check_fast_render.py` checks that every frame
is bit-identical to the official renderer. Evaluation (and the organizer)
always use the official renderer.
"""
import types
from functools import lru_cache

import numpy as np
import pygame
from gymnasium.envs.box2d import car_racing
from gymnasium.envs.box2d.car_racing import GRASS_DIM, MAX_SHAPE_DIM, WINDOW_H, WINDOW_W
from pygame import gfxdraw


def install(env: car_racing.CarRacing):
    """Patch one (unwrapped) CarRacing instance. The font cache is process-wide."""
    # Same grass squares, same order as CarRacing._render_road.
    grass = []
    for x in range(-20, 20, 2):
        for y in range(-20, 20, 2):
            grass.append([
                (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
            ])
    env._fast_grass = np.array(grass, dtype=np.float64)
    env._fast_road_src = None
    env._render_road = types.MethodType(_render_road, env)
    car_racing.pygame = _PygameCachedFont()


def _visible_screen_polys(polys: np.ndarray, zoom, translation, cos, sin):
    """(P, 4, 2) world polygons -> (screen polygons, mask of the ones CarRacing draws)."""
    x, y = polys[..., 0], polys[..., 1]
    # Same operations, in the same order, as Vector2.rotate_rad followed by
    # `c * zoom + translation` in CarRacing._draw_colored_polygon.
    sx = (cos * x - sin * y) * zoom + translation[0]
    sy = (sin * x + cos * y) * zoom + translation[1]
    inside = (
        (-MAX_SHAPE_DIM <= sx) & (sx <= WINDOW_W + MAX_SHAPE_DIM)
        & (-MAX_SHAPE_DIM <= sy) & (sy <= WINDOW_H + MAX_SHAPE_DIM)
    )
    return np.stack([sx, sy], axis=-1), inside.any(axis=1)


def _render_road(self, zoom, translation, angle):
    bounds = car_racing.PLAYFIELD
    field = [(bounds, bounds), (bounds, -bounds), (-bounds, -bounds), (-bounds, bounds)]
    self._draw_colored_polygon(self.surf, field, self.bg_color, zoom, translation, angle, clip=False)

    # cos and sin exactly as pygame computes them inside rotate_rad (including
    # its exact 0/90/180/270 degree special cases): rotate the unit vector.
    cos, sin = pygame.math.Vector2(1.0, 0.0).rotate_rad(angle)

    screen, visible = _visible_screen_polys(self._fast_grass, zoom, translation, cos, sin)
    for i in np.flatnonzero(visible):
        poly = [tuple(p) for p in screen[i].tolist()]
        gfxdraw.aapolygon(self.surf, poly, self.grass_color)
        gfxdraw.filled_polygon(self.surf, poly, self.grass_color)

    # road_poly is rebuilt at every reset. Its colors are not cached: they are
    # live arrays that change when a tile is first driven over.
    if self._fast_road_src is not self.road_poly:
        self._fast_road_src = self.road_poly
        self._fast_road = np.array([poly for poly, _ in self.road_poly], dtype=np.float64)
    screen, visible = _visible_screen_polys(self._fast_road, zoom, translation, cos, sin)
    for i in np.flatnonzero(visible):
        poly = [tuple(p) for p in screen[i].tolist()]
        color = [int(c) for c in self.road_poly[i][1]]
        gfxdraw.aapolygon(self.surf, poly, color)
        gfxdraw.filled_polygon(self.surf, poly, color)


class _FontCache:
    def __getattr__(self, name):
        return getattr(pygame.font, name)

    @staticmethod
    @lru_cache(maxsize=None)
    def Font(*args):
        return pygame.font.Font(*args)


class _PygameCachedFont:
    """Stands in for `pygame` inside car_racing.py only, with a cached Font."""

    font = _FontCache()

    def __getattr__(self, name):
        return getattr(pygame, name)
