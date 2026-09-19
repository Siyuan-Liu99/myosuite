# Adapted from Holosoma. Copyright Amazon.com, Inc. or its affiliates.
# Licensed under the Apache License, Version 2.0; see NOTICE.md.
"""Flax MLPs and squashed Gaussian policy used by FastSAC."""

import math

from flax import linen as nn
import jax
import jax.numpy as jp


class MLP(nn.Module):
    hidden_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        for width in (self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 4):
            x = nn.Dense(width)(x)
            if self.layer_norm:
                x = nn.LayerNorm(epsilon=1e-5)(x)
            x = nn.silu(x)
        return x


class Actor(nn.Module):
    action_size: int
    hidden_dim: int = 512
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs):
        x = MLP(self.hidden_dim, self.layer_norm)(obs)
        mean = nn.Dense(
            self.action_size, kernel_init=nn.initializers.zeros, name="mean"
        )(x)
        log_std = nn.Dense(
            self.action_size, kernel_init=nn.initializers.zeros, name="log_std"
        )(x)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            jp.tanh(log_std) + 1.0
        )
        return mean, log_std


class DistributionalCritic(nn.Module):
    num_atoms: int = 101
    hidden_dim: int = 768
    num_q_networks: int = 2
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs, action):
        x = jp.concatenate((obs, action), axis=-1)
        logits = []
        for i in range(self.num_q_networks):
            hidden = MLP(self.hidden_dim, self.layer_norm, name=f"q{i}")(x)
            logits.append(nn.Dense(self.num_atoms, name=f"head{i}")(hidden))
        return jp.stack(logits)  # (num_q_networks, batch, atoms)


def sample_action(actor, params, obs, rng):
    mean, log_std = actor.apply({"params": params}, obs)
    noise = jax.random.normal(rng, mean.shape)
    raw = mean + jp.exp(log_std) * noise
    action = jp.tanh(raw)
    log_prob = -0.5 * (noise**2 + 2.0 * log_std + math.log(2.0 * math.pi))
    # Stable log(1 - tanh(raw)**2), including near-saturated actions.
    log_prob -= 2.0 * (math.log(2.0) - raw - jax.nn.softplus(-2.0 * raw))
    return action, jp.sum(log_prob, axis=-1), jp.mean(jp.exp(log_std))


def deterministic_action(actor, params, obs):
    mean, _ = actor.apply({"params": params}, obs)
    return jp.tanh(mean)
