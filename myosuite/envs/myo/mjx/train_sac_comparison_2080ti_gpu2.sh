#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"
. ../../../../.venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# RTX 2080 Ti GPU 2: 5 sequential runs; stop if a run fails.
# One seed (42), 100M total transitions per run, W&B project myosuite.

python -u train_jax_sac.py --env_name=MjxChallengeDieReorientP2-v0 --impl=warp \
  --num_envs=64 --num_timesteps=100000000 --seed=42 --batch_size=256 \
  --min_replay_size=8192 --max_replay_size=262144 --grad_updates_per_step=1 \
  --reward_scaling=0.01 --num_evals=21 --num_eval_envs=64 --deterministic_eval \
  --log_interval=100 --log_to_wandb --wandb_project=myosuite --save_policy

python -u train_jax_fastsac.py --env_name=MjxChallengeDieReorientP2-v0 --impl=warp \
  --num_envs=64 --num_timesteps=100000000 --seed=42 --batch_size=256 \
  --warp_graph_cache_size=1 --buffer_size=4096 --learning_starts=128 --num_updates=8 --policy_frequency=4 \
  --reward_scale=0.01 --num_evals=21 --num_eval_envs=64 --save_interval=78125 \
  --log_interval=100 --log_to_wandb --wandb_project=myosuite

python -u train_jax_sac.py --env_name=MjxHandReorient100-v0 --impl=warp \
  --num_envs=64 --num_timesteps=100000000 --seed=42 --batch_size=256 \
  --min_replay_size=8192 --max_replay_size=262144 --grad_updates_per_step=1 \
  --reward_scaling=0.01 --num_evals=21 --num_eval_envs=64 --deterministic_eval \
  --log_interval=100 --log_to_wandb --wandb_project=myosuite --save_policy

python -u train_jax_fastsac.py --env_name=MjxHandReorient100-v0 --impl=warp \
  --num_envs=64 --num_timesteps=100000000 --seed=42 --batch_size=256 \
  --warp_graph_cache_size=1 --buffer_size=4096 --learning_starts=128 --num_updates=8 --policy_frequency=4 \
  --reward_scale=0.01 --num_evals=21 --num_eval_envs=64 --save_interval=78125 \
  --log_interval=100 --log_to_wandb --wandb_project=myosuite

python -u train_jax_fastsac.py --env_name=MjxChallengeDieReorientP1-v0 --impl=warp \
  --num_envs=64 --num_timesteps=100000000 --seed=42 --batch_size=256 \
  --warp_graph_cache_size=1 --buffer_size=4096 --learning_starts=128 --num_updates=8 --policy_frequency=4 \
  --reward_scale=0.01 --num_evals=21 --num_eval_envs=64 --save_interval=78125 \
  --log_interval=100 --log_to_wandb --wandb_project=myosuite

