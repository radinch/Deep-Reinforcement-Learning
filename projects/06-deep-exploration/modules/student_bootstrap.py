"""Section D of the homework: Bootstrapped DQN with randomized priors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from bootstrap_dqn import BootstrappedDQN


def _mlp(in_size: int, out_size: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_size
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    layers.append(nn.Linear(width, out_size))
    return nn.Sequential(*layers)


class RandomizedPriorHead(nn.Module):
    """A trainable Q-network plus an independent frozen randomized prior."""

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        prior_scale: float,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.trainable = _mlp(obs_size, num_actions, hidden_sizes)
        self.prior = _mlp(obs_size, num_actions, hidden_sizes)
        self.prior_scale = float(prior_scale)
        for parameter in self.prior.parameters():
            parameter.requires_grad_(False)
        self.prior.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Detaching is redundant because prior parameters are frozen, but makes the
        # fixed-function intent explicit while preserving gradients through obs.
        return self.trainable(obs) + self.prior_scale * self.prior(obs)


def build_bootstrap_ensemble(
    obs_size: int,
    num_actions: int,
    num_heads: int,
    prior_scale: float,
    hidden_sizes: tuple[int, ...],
) -> "nn.ModuleList":
    """Build K independent randomized-prior Q heads."""
    return nn.ModuleList(
        [
            RandomizedPriorHead(
                obs_size, num_actions, prior_scale, hidden_sizes
            )
            for _ in range(num_heads)
        ]
    )


def boot_sample_head(agent: "BootstrappedDQN", rng: "np.random.Generator") -> int:
    """Uniformly sample one acting head for an episode."""
    return int(rng.integers(agent.cfg.num_heads))


def boot_select_action(agent: "BootstrappedDQN", obs: "np.ndarray", head: int) -> int:
    """Choose the greedy action under the selected head."""
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.ensemble[head](obs_tensor)
    return int(q_values.argmax(dim=1).item())


def boot_eval_action(agent: "BootstrappedDQN", obs: "np.ndarray") -> int:
    """Choose greedily under ensemble-mean Q-values."""
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q_values = torch.stack([head(obs_tensor) for head in agent.ensemble], dim=0)
        mean_q = q_values.mean(dim=0)
    return int(mean_q.argmax(dim=1).item())


def boot_head_target(
    agent: "BootstrappedDQN",
    online_k: "nn.Module",
    target_k: "nn.Module",
    reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """Build one head's Double-DQN target."""
    with torch.no_grad():
        next_action = online_k(next_obs).argmax(dim=1, keepdim=True)
        next_q = target_k(next_obs).gather(1, next_action).squeeze(1)
        return reward + agent.cfg.gamma * (1.0 - done) * next_q


def boot_head_loss(
    agent: "BootstrappedDQN",
    online_k: "nn.Module",
    target_k: "nn.Module",
    obs: "torch.Tensor",
    action: "torch.Tensor",
    reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":
    """Return mean masked Huber loss for one ensemble head."""
    q_taken = online_k(obs).gather(1, action.long().unsqueeze(1)).squeeze(1)
    target = boot_head_target(agent, online_k, target_k, reward, next_obs, done)
    per_item = F.smooth_l1_loss(q_taken, target, reduction="none")
    return (per_item * mask).sum() / mask.sum().clamp_min(1.0)
