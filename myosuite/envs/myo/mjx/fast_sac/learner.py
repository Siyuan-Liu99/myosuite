# Adapted from Holosoma. Copyright Amazon.com, Inc. or its affiliates.
# Licensed under the Apache License, Version 2.0; see NOTICE.md.
"""Distributional FastSAC updates in JAX, independent of the simulator."""

import math

from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jp
import optax

from myosuite.envs.myo.mjx.fast_sac.networks import (
    Actor,
    DistributionalCritic,
    sample_action,
)


@struct.dataclass
class Normalizer:
    mean: jax.Array
    variance: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, obs_size):
        return cls(jp.zeros(obs_size), jp.ones(obs_size), jp.array(0.0))

    def update(self, observations):
        # Parallel Welford update, using each collected observation once.
        batch_count = observations.shape[0]
        count = self.count + batch_count
        delta = jp.mean(observations, axis=0) - self.mean
        mean = self.mean + delta * batch_count / count
        m2 = (
            self.variance * self.count
            + jp.var(observations, axis=0) * batch_count
            + delta**2 * self.count * batch_count / count
        )
        return self.replace(
            mean=mean, variance=jp.maximum(m2 / count, 0.0), count=count
        )

    def normalize(self, obs):
        return (obs - self.mean) / (jp.sqrt(self.variance) + 1e-2)


@struct.dataclass
class LearnerState:
    actor: TrainState
    critic: TrainState
    target_params: object
    log_alpha: jax.Array
    alpha_optimizer: object
    updates: jax.Array


def project_distribution(probabilities, target_atoms, v_min, v_max):
    """C51 projection; exact atom hits retain all their probability mass."""
    num_atoms = probabilities.shape[-1]
    coordinates = (jp.clip(target_atoms, v_min, v_max) - v_min) * (
        (num_atoms - 1) / (v_max - v_min)
    )
    lower = jp.clip(jp.floor(coordinates).astype(jp.int32), 0, num_atoms - 1)
    upper = jp.minimum(lower + 1, num_atoms - 1)
    upper_weight = jp.clip(coordinates - lower, 0.0, 1.0)

    def row(probs, lo, hi, weight):
        result = jp.zeros_like(probs)
        result = result.at[lo].add(probs * (1.0 - weight))
        return result.at[hi].add(probs * weight)

    per_batch = jax.vmap(row)
    return jax.vmap(per_batch, in_axes=(0, None, None, None))(
        probabilities, lower, upper, upper_weight
    )


class FastSAC:
    def __init__(self, config, obs_size, action_size):
        self.config = config
        self.actor = Actor(
            action_size,
            config.actor_hidden_dim,
            config.log_std_min,
            config.log_std_max,
            config.layer_norm,
        )
        self.critic = DistributionalCritic(
            config.num_atoms,
            config.critic_hidden_dim,
            config.num_q_networks,
            config.layer_norm,
        )
        self.obs_size = obs_size
        self.action_size = action_size
        self.support = jp.linspace(config.v_min, config.v_max, config.num_atoms)
        self.target_entropy = -action_size * config.target_entropy_ratio
        # Match the upstream alpha AdamW default weight decay separately from
        # the explicitly configured actor/critic weight decay.
        self.alpha_tx = optax.adamw(
            config.alpha_learning_rate, b1=0.9, b2=0.95, weight_decay=0.01
        )

    def _optimizer(self, learning_rate):
        transforms = []
        if self.config.max_grad_norm > 0:
            transforms.append(optax.clip_by_global_norm(self.config.max_grad_norm))
        transforms.append(
            optax.adamw(
                learning_rate,
                b1=0.9,
                b2=0.95,
                weight_decay=self.config.weight_decay,
            )
        )
        return optax.chain(*transforms)

    def init(self, rng):
        actor_rng, critic_rng = jax.random.split(rng)
        obs = jp.zeros((1, self.obs_size))
        action = jp.zeros((1, self.action_size))
        actor_params = self.actor.init(actor_rng, obs)["params"]
        critic_params = self.critic.init(critic_rng, obs, action)["params"]
        log_alpha = jp.array(math.log(self.config.alpha_init), dtype=jp.float32)
        return LearnerState(
            actor=TrainState.create(
                apply_fn=self.actor.apply,
                params=actor_params,
                tx=self._optimizer(self.config.actor_learning_rate),
            ),
            critic=TrainState.create(
                apply_fn=self.critic.apply,
                params=critic_params,
                tx=self._optimizer(self.config.critic_learning_rate),
            ),
            target_params=critic_params,
            log_alpha=log_alpha,
            alpha_optimizer=self.alpha_tx.init(log_alpha),
            updates=jp.array(0, dtype=jp.int32),
        )

    def update(self, state, batch, rng):
        next_rng, policy_rng = jax.random.split(rng)
        next_action, next_log_prob, _ = sample_action(
            self.actor, state.actor.params, batch["next_obs"], next_rng
        )
        target_logits = self.critic.apply(
            {"params": state.target_params}, batch["next_obs"], next_action
        )
        target_probs = jax.nn.softmax(target_logits, axis=-1)
        target_atoms = batch["reward"][:, None] + batch["discount"][:, None] * (
            self.support[None, :] - jp.exp(state.log_alpha) * next_log_prob[:, None]
        )
        target = jax.lax.stop_gradient(
            project_distribution(
                target_probs, target_atoms, self.config.v_min, self.config.v_max
            )
        )

        def critic_loss(params):
            logits = self.critic.apply(
                {"params": params}, batch["obs"], batch["action"]
            )
            # Upstream trains each critic against its OWN target distribution;
            # do not silently substitute the minimum-Q target from ordinary SAC.
            cross_entropy = -jp.sum(
                target * jax.nn.log_softmax(logits, axis=-1), axis=-1
            )
            return jp.sum(jp.mean(cross_entropy, axis=1))

        q_loss, q_grad = jax.value_and_grad(critic_loss)(state.critic.params)
        state = state.replace(critic=state.critic.apply_gradients(grads=q_grad))
        alpha_loss = jp.array(0.0)
        if self.config.autotune:

            def temperature_loss(log_alpha):
                return -jp.mean(
                    jp.exp(log_alpha)
                    * jax.lax.stop_gradient(next_log_prob + self.target_entropy)
                )

            alpha_loss, alpha_grad = jax.value_and_grad(temperature_loss)(
                state.log_alpha
            )
            alpha_update, alpha_optimizer = self.alpha_tx.update(
                alpha_grad, state.alpha_optimizer, state.log_alpha
            )
            state = state.replace(
                log_alpha=optax.apply_updates(state.log_alpha, alpha_update),
                alpha_optimizer=alpha_optimizer,
            )

        def update_actor(actor_state):
            def actor_loss(params):
                actions, log_probs, action_std = sample_action(
                    self.actor, params, batch["obs"], policy_rng
                )
                logits = self.critic.apply(
                    {"params": state.critic.params}, batch["obs"], actions
                )
                q_values = jp.sum(
                    jax.nn.softmax(logits, axis=-1) * self.support, axis=-1
                )
                # Like Holosoma, optimize the mean of the ensemble (not min).
                loss = jp.mean(
                    jp.exp(state.log_alpha) * log_probs - jp.mean(q_values, axis=0)
                )
                return loss, (-jp.mean(log_probs), action_std)

            (loss, (entropy, std)), grad = jax.value_and_grad(actor_loss, has_aux=True)(
                actor_state.params
            )
            return actor_state.apply_gradients(grads=grad), (
                loss,
                entropy,
                std,
                optax.global_norm(grad),
            )

        policy_update = (state.updates + 1) % self.config.policy_frequency == 0
        actor, actor_metrics = jax.lax.cond(
            policy_update,
            update_actor,
            lambda x: (x, (jp.array(0.0),) * 4),
            state.actor,
        )
        state = state.replace(
            actor=actor,
            target_params=optax.incremental_update(
                state.critic.params, state.target_params, self.config.tau
            ),
            updates=state.updates + 1,
        )
        clipped = (target_atoms < self.config.v_min) | (
            target_atoms > self.config.v_max
        )
        target_values = jp.sum(target * self.support, axis=-1)
        metrics = {
            "critic_loss": q_loss,
            "critic_grad_norm": optax.global_norm(q_grad),
            "alpha_loss": alpha_loss,
            "alpha": jp.exp(state.log_alpha),
            "target_q_min": jp.min(target_values),
            "target_q_max": jp.max(target_values),
            "support_clip_fraction": jp.mean(jp.sum(target_probs * clipped, axis=-1)),
            "actor_loss": actor_metrics[0],
            "policy_entropy": actor_metrics[1],
            "action_std": actor_metrics[2],
            "actor_grad_norm": actor_metrics[3],
            "actor_updates": policy_update.astype(jp.float32),
        }
        return state, metrics

    def update_many(self, state, batches, rng):
        keys = jax.random.split(rng, self.config.num_updates)

        def step(state, inputs):
            batch, key = inputs
            return self.update(state, batch, key)

        state, values = jax.lax.scan(step, state, (batches, keys))
        metrics = jax.tree.map(jp.mean, values)
        # Do not dilute actor metrics with zero values on critic-only updates.
        count = jp.sum(values["actor_updates"])
        for key in ("actor_loss", "policy_entropy", "action_std", "actor_grad_norm"):
            metrics[key] = jp.sum(values[key]) / jp.maximum(count, 1)
        metrics["actor_updates"] = count
        return state, metrics
