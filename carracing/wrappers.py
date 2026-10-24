"""Environment wrappers used during training.

The agent never sees these at evaluation time: the official script hands raw
RGB frames to the model. `driver.py` reproduces the same preprocessing
(grayscale + frame stack + action repeat) inside the submitted model.
"""
from collections import deque

import gymnasium as gym
import numpy as np

FRAME_STACK = 4
ACTION_REPEAT = 4

# The car is always drawn at the same place in the 96x96 frame.
CAR_ROWS = slice(66, 78)
CAR_COLS = slice(44, 52)
# CarRacing zooms in during its first second (50 frames): the car is not yet
# at its final position, so the grass check is skipped then.
ZOOM_FRAMES = 50


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


def action_to_env(action: np.ndarray) -> np.ndarray:
    """Policy action in [0, 1]^2 -> env action [steer, gas, brake].

    action[0] -> steer in [-1, 1]
    action[1] -> throttle in [-1, 1]: positive part is gas, negative is brake,
                 so the agent can never press both pedals at once.
    """
    steer = action[0] * 2.0 - 1.0
    throttle = action[1] * 2.0 - 1.0
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
        grass_penalty: float = 0.5,
        patience: int = 50,
        stuck_penalty: float = 20.0,
    ):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.patience = patience
        self.stuck_penalty = stuck_penalty
        self.frames = deque(maxlen=FRAME_STACK)
        self.observation_space = gym.spaces.Box(
            0, 255, (FRAME_STACK, 96, 96), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(0.0, 1.0, (2,), dtype=np.float32)

    def reset(self, **kwargs):
        rgb, info = self.env.reset(**kwargs)
        self.frame_count = 0
        self.steps_without_progress = 0
        self.raw_return = 0.0
        gray = to_gray(rgb)
        for _ in range(FRAME_STACK):
            self.frames.append(gray)
        return np.stack(self.frames), info

    def step(self, action):
        env_action = action_to_env(action)
        total_reward = 0.0
        shaped_reward = 0.0
        for _ in range(ACTION_REPEAT):
            rgb, reward, terminated, truncated, info = self.env.step(env_action)
            self.frame_count += 1
            total_reward += reward
            if terminated or truncated:
                break

        # Reward shaping (training only).
        if self.frame_count > ZOOM_FRAMES and on_grass(rgb):
            shaped_reward -= self.grass_penalty

        # A new tile gives ~+3.5, so a non-positive reward means no progress.
        if total_reward > 0:
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
        if self.steps_without_progress >= self.patience and not truncated:
            # Without this penalty, ending the episode early would look
            # attractive: every remaining step would only cost -0.1 per frame.
            shaped_reward -= self.stuck_penalty
            terminated = True

        # The unshaped score is kept so the logs show the real game score.
        self.raw_return += total_reward
        if terminated or truncated:
            info["episode_raw_return"] = self.raw_return
            info["episode_frames"] = self.frame_count
        self.frames.append(to_gray(rgb))
        return (
            np.stack(self.frames),
            total_reward + shaped_reward,
            terminated,
            truncated,
            info,
        )


def make_env(max_episode_steps: int = 3000, **wrapper_kwargs) -> gym.Env:
    env = gym.make("CarRacing-v3", max_episode_steps=max_episode_steps)
    return CarRacingWrapper(env, **wrapper_kwargs)
