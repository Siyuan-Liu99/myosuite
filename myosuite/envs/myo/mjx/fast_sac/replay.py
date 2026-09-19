# Adapted from Holosoma. Copyright Amazon.com, Inc. or its affiliates.
# Licensed under the Apache License, Version 2.0; see NOTICE.md.
"""Device-resident replay with episode-safe n-step sampling."""

from flax import struct
import jax
import jax.numpy as jp


@struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    terminated: jax.Array
    truncated: jax.Array


@struct.dataclass
class ReplayState:
    data: Transition
    position: jax.Array
    size: jax.Array


def init_replay(capacity, num_envs, obs_size, action_size):
    """Capacity is control steps PER ENV, not total transitions."""
    return ReplayState(
        Transition(
            obs=jp.zeros((capacity, num_envs, obs_size)),
            action=jp.zeros((capacity, num_envs, action_size)),
            reward=jp.zeros((capacity, num_envs)),
            next_obs=jp.zeros((capacity, num_envs, obs_size)),
            terminated=jp.zeros((capacity, num_envs), dtype=bool),
            truncated=jp.zeros((capacity, num_envs), dtype=bool),
        ),
        position=jp.array(0, dtype=jp.int32),
        size=jp.array(0, dtype=jp.int32),
    )


def insert(replay, transition):
    capacity = replay.data.reward.shape[0]
    return replay.replace(
        data=jax.tree.map(
            lambda x, y: x.at[replay.position].set(y), replay.data, transition
        ),
        position=(replay.position + 1) % capacity,
        size=jp.minimum(replay.size + 1, capacity),
    )


def sample(replay, rng, batch_size, n_steps, gamma):
    """Sample consecutive transitions without crossing a reset or ring seam.

    Termination ends both the return and bootstrap. Time truncation ends the
    return but bootstraps from the FINAL observation of that same episode.
    The caller must fill at least n_steps rows before sampling.
    """
    capacity, num_envs = replay.data.reward.shape
    env_rng, time_rng = jax.random.split(rng)
    env_ids = jax.random.randint(env_rng, (batch_size,), 0, num_envs)
    offsets = jax.random.randint(time_rng, (batch_size,), 0, replay.size - n_steps + 1)
    oldest = jp.where(replay.size == capacity, replay.position, 0)
    rows = (oldest + offsets[:, None] + jp.arange(n_steps)[None, :]) % capacity
    seq = jax.tree.map(lambda x: x[rows, env_ids[:, None]], replay.data)
    boundary = seq.terminated | seq.truncated
    alive = jp.cumprod(
        jp.concatenate(
            (jp.ones((batch_size, 1)), (~boundary[:, :-1]).astype(jp.float32)), axis=1
        ),
        axis=1,
    )
    lengths = jp.sum(alive, axis=1).astype(jp.int32)
    final = lengths - 1
    batch_ids = jp.arange(batch_size)
    reward = jp.sum(seq.reward * alive * gamma ** jp.arange(n_steps), axis=1)
    discount = gamma**lengths * (~seq.terminated[batch_ids, final])
    return {
        "obs": seq.obs[:, 0],
        "action": seq.action[:, 0],
        "reward": reward,
        "next_obs": seq.next_obs[batch_ids, final],
        "discount": discount,
    }
