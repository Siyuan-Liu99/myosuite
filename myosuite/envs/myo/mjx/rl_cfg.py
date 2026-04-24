from ml_collections import config_dict
from jax import numpy as jp


ppo_config = config_dict.create(
    num_timesteps=10_000_000,
    learning_rate=3e-4,
    discounting=0.97,
    gae_lambda=0.95,
    entropy_cost=0.001,
    clipping_epsilon=0.3,
    max_grad_norm=1.0,
    action_repeat=1,
    num_minibatches=32,
    num_updates_per_batch=8,
    batch_size=256,
    unroll_length=10,
    reward_scaling=1.0,
    normalize_observations=True,
    num_evals=16,
    num_eval_envs=128,
    num_resets_per_eval=1,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(64, 64, 64),
        value_hidden_layer_sizes=(64, 64, 64),
        policy_obs_key="state",
        value_obs_key="state",
    ),
)

sac_config = config_dict.create(
    num_timesteps=100_000_000,
    learning_rate=3e-4,
    discounting=0.97,
    batch_size=256,
    num_evals=16,
    num_eval_envs=128,
    action_repeat=1,
    normalize_observations=True,
    reward_scaling=1.0,
    tau=0.005,
    min_replay_size=8192,
    max_replay_size=1_000_000,
    grad_updates_per_step=1,
    deterministic_eval=False,
    seed=0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(256, 256),
        q_hidden_layer_sizes=(256, 256),
    ),
)
