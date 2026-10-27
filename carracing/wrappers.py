"""Environment wrappers used during training.

The agent never sees these at evaluation time: the official script hands raw
RGB frames to the model. `driver.py` reproduces the same preprocessing
(grayscale + frame stack + action repeat) inside the submitted model.
"""
from collections import deque

import gymnasium as gym
import numpy as np

from carracing.fast_render import install as install_fast_render

FRAME_STACK = 4
ACTION_REPEAT = 4

# The car is always drawn at the same place in the 96x96 frame.
CAR_ROWS = slice(66, 78)
CAR_COLS = slice(44, 52)
# CarRacing zooms in during its first second (50 frames): the car is not yet
# at its final position, so the grass check is skipped then.
ZOOM_FRAMES = 50
# Skipped-tile check: the furthest tile reached is searched this many tiles
# ahead (further jumps are the track passing close to itself, not progress),
# and a tile counts as missed once the car is this many tiles past it.
TILE_LOOKAHEAD = 30
MISS_MARGIN = 8


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """(96, 96, 3) uint8 RGB -> (96, 96) uint8 grayscale."""
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return gray.astype(np.uint8)


def on_grass(rgb: np.ndarray) -> bool:
    """True when most pixels around the car are grass green (~[100, 202, 100])."""
    patch = rgb[CAR_ROWS, CAR_COLS].astype(np.int16)
    green = (patch[..., 1] > 150) & (patch[..., 0] < 150)
    # The car itself covers ~40% of the patch, so ~0.6 means fully on grass.
    return green.mean() > 0.5


# The policy outputs Beta samples, and the driver uses the Beta mean, which
# never reaches 0 or 1. With a plain linear mapping, full gas needed exactly
# action[1] = 1, so v0 never went above 0.76 gas. With this gain the pedals
# saturate before the edge: full gas at action[1] >= 0.85, full brake <= 0.15.
THROTTLE_GAIN = 1.0 / 0.7


def action_to_env(action: np.ndarray) -> np.ndarray:
    """Policy action in [0, 1]^2 -> env action [steer, gas, brake].

    action[0] -> steer in [-1, 1]
    action[1] -> throttle in [-1, 1]: positive part is gas, negative is brake,
                 so the agent can never press both pedals at once.
    """
    steer = action[0] * 2.0 - 1.0
    throttle = float(np.clip((action[1] * 2.0 - 1.0) * THROTTLE_GAIN, -1.0, 1.0))
    gas = max(throttle, 0.0)
    brake = max(-throttle, 0.0)
    return np.array([steer, gas, brake], dtype=np.float32)


class CarRacingWrapper(gym.Wrapper):
    """Action repeat + grayscale + frame stack + reward shaping.

    Observation: (4, 96, 96) uint8, oldest frame first.
    Action: (2,) in [0, 1] (see `action_to_env`).
    """

    def __init__(
        self,
        env: gym.Env,
        # Reward shaping, all off by default since v3: the env reward alone is
        # +1000/N per new tile and -0.1 per frame, i.e. exactly the lap time
        # once every tile is driven over. v1/v2 used the values in their
        # checkpoint args (0.5, 50, 20, 10, 0.2).
        grass_penalty: float = 0.0,
        patience: int = 0,  # 0 = never end an episode early
        stuck_penalty: float = 20.0,
        missed_tile_penalty: float = 0.0,
        steer_penalty: float = 0.0,
        fast_render: bool = True,
    ):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.patience = patience
        self.stuck_penalty = stuck_penalty
        self.missed_tile_penalty = missed_tile_penalty
        self.steer_penalty = steer_penalty
        self.frames = deque(maxlen=FRAME_STACK)
        self.observation_space = gym.spaces.Box(
            0, 255, (FRAME_STACK, 96, 96), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(0.0, 1.0, (2,), dtype=np.float32)

        # CarRacing draws a frame on every physics step, and drawing costs
        # ~80% of the step time. Only the last of the ACTION_REPEAT frames is
        # observed, so the others are skipped. Drawing has no effect on the
        # physics or the reward, so the episode is exactly the same.
        car_racing = env.unwrapped
        if fast_render:
            install_fast_render(car_racing)
        self._draw = car_racing._render
        self.skip_draw = False
        car_racing._render = lambda mode: car_racing.state if self.skip_draw else self._draw(mode)

    def reset(self, **kwargs):
        rgb, info = self.env.reset(**kwargs)
        self.frame_count = 0
        self.steps_without_progress = 0
        self.raw_return = 0.0
        self.furthest_tile = 0  # the car spawns on tile 0
        self.checked_tile = 0
        self.missed_tiles = 0
        self.last_steer = 0.0
        gray = to_gray(rgb)
        for _ in range(FRAME_STACK):
            self.frames.append(gray)
        return np.stack(self.frames), info

    def step(self, action):
        env_action = action_to_env(action)
        total_reward = 0.0
        shaped_reward = 0.0
        for i in range(ACTION_REPEAT):
            self.skip_draw = i < ACTION_REPEAT - 1
            rgb, reward, terminated, truncated, info = self.env.step(env_action)
            self.frame_count += 1
            total_reward += reward
            if terminated or truncated:
                break
        if self.skip_draw:  # the episode ended on a skipped frame
            rgb = self._draw("state_pixels")
        self.skip_draw = False

        # Reward shaping (training only).
        if self.frame_count > ZOOM_FRAMES and on_grass(rgb):
            shaped_reward -= self.grass_penalty

        # Tile 0 is already visited at spawn, so CarRacing's "95% is a lap"
        # rule never fires: the lap ends only when every tile is touched. One
        # tile skipped by cutting a corner costs a whole extra lap (~15 s),
        # far beyond what gamma sees, so it is penalized as soon as it happens.
        # Once the last tile is reached, furthest_tile counts on past the finish
        # (tile n = tile 0, already visited): otherwise the last MISS_MARGIN
        # tiles, i.e. the final corner, were never checked, and v2 learned to
        # cut it. Before that, the search stops at the last tile, or the
        # visited first tiles would look like progress while still far away.
        road = self.env.unwrapped.road
        n = len(road)
        end = self.furthest_tile + TILE_LOOKAHEAD
        if self.furthest_tile < n - 1:
            end = min(end, n)
        for j in range(self.furthest_tile + 1, end):
            if road[j % n].road_visited:
                self.furthest_tile = j
        while self.checked_tile < min(self.furthest_tile - MISS_MARGIN, n - 1):
            self.checked_tile += 1
            if not road[self.checked_tile].road_visited:
                self.missed_tiles += 1
                shaped_reward -= self.missed_tile_penalty

        # v1 flipped its steering on ~40% of decisions and weaved on straights:
        # nothing made a jerk of the wheel cost anything.
        steer = float(env_action[0])
        shaped_reward -= self.steer_penalty * abs(steer - self.last_steer)
        self.last_steer = steer

        # A new tile gives ~+3.5, so a non-positive reward means no progress.
        if total_reward > 0:
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
        if self.patience and self.steps_without_progress >= self.patience and not truncated:
            # Without this penalty, ending the episode early would look
            # attractive: every remaining step would only cost -0.1 per frame.
            shaped_reward -= self.stuck_penalty
            terminated = True

        # The unshaped score is kept so the logs show the real game score.
        self.raw_return += total_reward
        if terminated or truncated:
            info["episode_raw_return"] = self.raw_return
            info["episode_frames"] = self.frame_count
            info["episode_missed_tiles"] = self.missed_tiles
        self.frames.append(to_gray(rgb))
        return (
            np.stack(self.frames),
            total_reward + shaped_reward,
            terminated,
            truncated,
            info,
        )


def make_env(max_episode_steps: int = 3000, **wrapper_kwargs) -> gym.Env:
    # The checker only warns that skipped frames reuse the previous obs array.
    env = gym.make(
        "CarRacing-v3", max_episode_steps=max_episode_steps, disable_env_checker=True
    )
    return CarRacingWrapper(env, **wrapper_kwargs)
