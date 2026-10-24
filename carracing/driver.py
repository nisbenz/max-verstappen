"""The submitted model: raw RGB frame in, [steer, gas, brake] out.

The official evaluation script only gives the model one raw (96, 96, 3) frame
per env step and expects a continuous [steer, gas, brake] action back. During
training the wrappers did the grayscale, the frame stack, the action repeat
and the throttle -> gas/brake split. `Driver` does exactly the same things
inside the model, so the organizer only has to call:

    action = model(obs)
    obs, reward, terminated, truncated, info = env.step(action)
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from carracing.model import ActorCritic
from carracing.wrappers import ACTION_REPEAT, FRAME_STACK, action_to_env, to_gray


class Driver(nn.Module):
    def __init__(self, policy: ActorCritic | None = None):
        super().__init__()
        self.policy = policy if policy is not None else ActorCritic(FRAME_STACK)
        self.reset()

    def reset(self):
        """Forget the previous episode. A freshly loaded model is already reset."""
        self.frames = deque(maxlen=FRAME_STACK)
        self.step_count = 0
        self.last_action = np.zeros(3, dtype=np.float32)

    @torch.no_grad()
    def forward(self, obs) -> np.ndarray:
        # Action repeat: decide only every 4th frame, like during training.
        if self.step_count % ACTION_REPEAT == 0:
            gray = to_gray(np.asarray(obs, dtype=np.uint8))
            if not self.frames:
                self.frames.extend([gray] * FRAME_STACK)
            else:
                self.frames.append(gray)
            stack = torch.from_numpy(np.stack(self.frames)).unsqueeze(0)  # (1, 4, 96, 96)
            device = next(self.policy.parameters()).device
            action = self.policy.greedy_action(stack.to(device))[0].cpu().numpy()
            self.last_action = action_to_env(action)
        self.step_count += 1
        return self.last_action
