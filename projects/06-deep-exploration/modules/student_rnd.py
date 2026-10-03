"""Section C of the homework: Random Network Distillation (RND)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from rnd_dqn import RNDDoubleDQN


def _mlp(in_size: int, out_size: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_size
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    layers.append(nn.Linear(width, out_size))
    return nn.Sequential(*layers)


def build_rnd_networks(
    obs_size: int, output_dim: int, hidden_sizes: tuple[int, ...]
) -> tuple[nn.Module, nn.Module]:
    """Build a frozen random target and a trainable predictor MLP."""
    target = _mlp(obs_size, output_dim, hidden_sizes)
    predictor = _mlp(obs_size, output_dim, hidden_sizes)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()
    return target, predictor


def rnd_intrinsic_bonus(
    agent: "RNDDoubleDQN", next_obs: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Compute normalized per-state RND error and predictor MSE loss."""
    with torch.no_grad():
        target_embedding = agent.rnd_target(next_obs)
    predictor_embedding = agent.rnd_predictor(next_obs)

    # Mean squared embedding error gives one raw novelty value per transition.
    raw_error = (predictor_embedding - target_embedding).pow(2).mean(dim=1)
    predictor_loss = raw_error.mean()

    raw_detached = raw_error.detach()
    agent.intrinsic_rms.update(raw_detached.cpu().numpy())
    scale = max(float(agent.intrinsic_rms.std), 1e-8)
    intrinsic = raw_detached / scale
    return intrinsic, predictor_loss


def rnd_td_target(
    agent: "RNDDoubleDQN",
    shaped_reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """Build the Double-DQN target on shaped rewards."""
    with torch.no_grad():
        next_action = agent.online(next_obs).argmax(dim=1, keepdim=True)
        next_q = agent.target(next_obs).gather(1, next_action).squeeze(1)
        return shaped_reward + agent.cfg.gamma * (1.0 - done) * next_q


def rnd_q_loss(
    agent: "RNDDoubleDQN",
    obs: "torch.Tensor",
    action: "torch.Tensor",
    shaped_reward: "torch.Tensor",
    next_obs: "torch.Tensor",
    done: "torch.Tensor",
) -> "torch.Tensor":
    """Return Huber loss between Q(s,a) and the shaped Double-DQN target."""
    q_taken = agent.online(obs).gather(1, action.long().unsqueeze(1)).squeeze(1)
    target = rnd_td_target(agent, shaped_reward, next_obs, done)
    return F.smooth_l1_loss(q_taken, target)
