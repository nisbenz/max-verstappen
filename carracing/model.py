"""CNN actor-critic.

Input:  (B, 4, 96, 96) uint8 grayscale frames t-3, t-2, t-1, t.
Output: a Beta distribution per action dimension (actor) and V(s) (critic).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

N_ACTIONS = 2  # steer, throttle (see wrappers.action_to_env)


def layer_init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, n_stack: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(n_stack, 32, kernel_size=8, stride=4)),  # -> 32x23x23
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),  # -> 64x10x10
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),  # -> 64x8x8
            nn.ReLU(),
            nn.Flatten(),  # -> 4096
            layer_init(nn.Linear(64 * 8 * 8, 512)),
            nn.ReLU(),
        )
        # Small gain: every state starts at about Beta(1.7, 1.7), a wide bump
        # centred on "straight, no pedal", instead of a random strong preference.
        self.alpha_head = layer_init(nn.Linear(512, N_ACTIONS), gain=0.01)
        self.beta_head = layer_init(nn.Linear(512, N_ACTIONS), gain=0.01)
        self.value_head = layer_init(nn.Linear(512, 1), gain=1.0)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs.float() / 255.0)
        # softplus(.) + 1 keeps alpha, beta > 1, so the Beta is unimodal
        # (one preferred action) instead of U-shaped.
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        value = self.value_head(h).squeeze(-1)
        return alpha, beta, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)[2]

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """Sample an action (rollout) or score a stored one (PPO update)."""
        alpha, beta, value = self.forward(obs)
        dist = Beta(alpha, beta)
        if action is None:
            # A float32 sample can land exactly on 0 or 1, where log_prob is -inf.
            action = dist.sample().clamp(1e-6, 1 - 1e-6)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean of the Beta: the deterministic action used for evaluation."""
        alpha, beta, _ = self.forward(obs)
        return alpha / (alpha + beta)
