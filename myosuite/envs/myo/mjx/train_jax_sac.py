"""Train a SAC agent using brax."""

import functools
import time
from datetime import datetime
import jax

print(f"Current backend: {jax.default_backend()}. "
      f"If you expect gpu but see cpu, you may need to reinstall jax with a suitable version of cuda.")
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from myosuite.envs.myo.mjx.rl_cfg import sac_config

from myosuite.envs.myo.mjx import make, get_default_config
from mujoco_playground import wrapper
import pickle
import wandb
import argparse


def main(env_name, impl, log_to_wandb, save_policy, num_envs=4096):
    """Run training and evaluation for the specified environment."""

    env, sac_params, network_factory = load_env_and_network_factory(
        env_name, impl, num_envs
    )

    if log_to_wandb:
        run_name = f"{env_name}-{datetime.now().strftime('%m%d-%H%M')}"
        wandb.init(project="myosuite", name=run_name, config=sac_params)

    # Train the model
    make_inference_fn, params, _ = sac.train(
        environment=env,
        num_envs=env._config.num_envs,
        episode_length=env._config.max_episode_steps,
        progress_fn=functools.partial(progress, log_to_wandb=log_to_wandb),
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        **sac_params,
    )

    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")
    if save_policy:
        with open('playground_params.pickle', 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_env_and_network_factory(env_name, impl, num_envs=4096):
    env = make(env_name, config_overrides={"impl": impl, "num_envs": num_envs})
    config = get_default_config(env_name)
    config.update({"impl": impl, "num_envs": num_envs})
    sac_params = dict(sac_config)

    print(f"Training on environment:\n{env_name}")
    print(f"Using backend:\n{impl}")
    print(f"Environment Config:\n{config}")
    print(f"SAC Training Parameters:\n{sac_config}")

    if "network_factory" in sac_params:
        network_factory = functools.partial(
            sac_networks.make_sac_networks, **sac_params.pop("network_factory")
        )
    else:
        network_factory = sac_networks.make_sac_networks

    return env, sac_params, network_factory


times = [time.monotonic()]
total_steps = [0]


# Progress function for logging
def progress(num_steps, metrics, log_to_wandb):
    times.append(time.monotonic())
    total_steps.append(num_steps)
    print(
        f"Step {num_steps} at {times[-1]}: reward={metrics['eval/episode_reward']:.3f}"
    )
    if log_to_wandb:
        wandb.log(metrics, step=num_steps)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train SAC agent with Brax")
    parser.add_argument(
        "--env_name",
        type=str,
        default="MjxFingerPoseRandom-v0",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4096,
        help="Number of environments to run in parallel",
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="jax",
        help='Implementation to use: "jax" (MJX) or "warp" (MJWarp)',
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
    )
    parser.add_argument(
        "--save_policy",
        action="store_true",
    )

    args = parser.parse_args()

    main(args.env_name, args.impl, args.log_to_wandb, args.save_policy, args.num_envs)
