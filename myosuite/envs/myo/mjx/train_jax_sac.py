"""Train Brax SAC; defaults target one 11 GB GPU per experiment.

Argument parsing is standard-library-only: --help does not initialize CUDA.
"""

import argparse
from datetime import datetime
import functools
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    def add(name, **kwargs):
        flags = ["--" + name]
        if "_" in name:
            flags.append("--" + name.replace("_", "-"))
        p.add_argument(*flags, **kwargs)

    add("env_name", default="MjxHandKeyTurnFixed-v0")
    add("impl", choices=("jax", "warp"), default="warp")
    add("num_envs", type=int, default=256)
    add("num_timesteps", type=int, default=100_000_000)
    add("seed", type=int, default=42)
    add("batch_size", type=int, default=1024)
    add(
        "min_replay_size",
        type=int,
        default=8192,
        help="Total transitions collected before learning",
    )
    add(
        "max_replay_size",
        type=int,
        default=262_144,
        help="Total replay capacity across environments",
    )
    add("grad_updates_per_step", type=int, default=1)
    add(
        "num_evals",
        type=int,
        default=21,
        help="Evaluations including step zero when >=2",
    )
    add("num_eval_envs", type=int, default=128)
    add("deterministic_eval", action=argparse.BooleanOptionalAction, default=True)
    add(
        "reward_scaling",
        type=float,
        default=None,
        help="Auto: 0.01 for KeyTurn/Reorient/PenTwirl, otherwise 1.0; logs use raw rewards",
    )
    add("log_to_wandb", action="store_true")
    add("wandb_project", default="myosuite")
    add("wandb_entity", default=None)
    add("log_dir", default="runs/sac")
    add(
        "save_policy",
        action="store_true",
        help="Also save a final pickle inside the unique run directory",
    )
    return p


def validate(p, args):
    for name in (
        "num_envs",
        "num_timesteps",
        "batch_size",
        "min_replay_size",
        "max_replay_size",
        "grad_updates_per_step",
        "num_evals",
        "num_eval_envs",
    ):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if args.max_replay_size < max(args.min_replay_size, args.num_envs):
        p.error("max_replay_size must be >= min_replay_size and num_envs")
    prefill = math.ceil(args.min_replay_size / args.num_envs) * args.num_envs
    if args.num_timesteps <= prefill:
        p.error("num_timesteps must leave training steps after replay prefill")
    if not 0 <= args.seed < 2**32 - 1:
        p.error("seed must be between 0 and 2**32 - 2")
    if args.reward_scaling is None:
        name = args.env_name.lower()
        args.reward_scaling = (
            0.01 if any(s in name for s in ("keyturn", "reorient", "pentwirl")) else 1.0
        )
    if not math.isfinite(args.reward_scaling) or args.reward_scaling <= 0:
        p.error("reward_scaling must be finite and positive")


def load_env_and_network_factory(env_name, impl, num_envs=256, **overrides):
    from brax.training.agents.sac import networks as sac_networks
    from myosuite.envs.myo.mjx import make, get_default_config
    from myosuite.envs.myo.mjx.rl_cfg import sac_config
    from myosuite.envs.myo.mjx.training_wrappers import FlatStateObservationWrapper

    env = FlatStateObservationWrapper(
        make(env_name, config_overrides={"impl": impl, "num_envs": num_envs})
    )
    config = get_default_config(env_name)
    config.update({"impl": impl, "num_envs": num_envs})
    sac_params = sac_config.to_dict()
    sac_params.update(overrides)
    network_config = sac_params.pop("network_factory", {})
    network_factory = functools.partial(
        sac_networks.make_sac_networks, **network_config
    )
    print(f"Environment: {env_name}\nConfig: {config}\nSAC: {sac_params}", flush=True)
    return env, sac_params, network_factory


def main(
    env_name,
    impl,
    log_to_wandb,
    save_policy,
    num_envs=256,
    *,
    wandb_project="myosuite",
    wandb_entity=None,
    log_dir="runs/sac",
    **overrides,
):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    import jax
    from brax.training.agents.sac import train as sac
    from myosuite.envs.myo.mjx.manipulation_config import CPU_ALIASES
    from myosuite.envs.myo.mjx.rl_cfg import sac_config
    from myosuite.envs.myo.mjx.training_wrappers import wrap_for_training

    started = time.monotonic()
    env_name = CPU_ALIASES.get(env_name, env_name)
    stamp = datetime.now()
    run_name = f"{env_name}-sac-{stamp:%m%d-%H%M}"
    # Keep the requested short W&B name, but avoid overwriting local runs.
    run_dir = Path(log_dir).expanduser().resolve() / f"{run_name}-{stamp:%Y%S%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"JAX devices: {jax.devices()}\nRun directory: {run_dir}", flush=True)
    if impl == "warp" and jax.default_backend() != "gpu":
        raise RuntimeError("--impl=warp requires a GPU JAX backend")
    env, sac_params, network_factory = load_env_and_network_factory(
        env_name, impl, num_envs, **overrides
    )
    metadata = {
        "algorithm": "sac",
        "env_name": env_name,
        "impl": impl,
        "num_envs": num_envs,
        "episode_length": env._config.max_episode_steps,
        **sac_params,
        "network_factory": sac_config.network_factory.to_dict(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    # Brax stores a flattened float32 transition, including done/truncation.
    replay_bytes = (
        sac_params["max_replay_size"]
        * 4
        * (2 * env.observation_size + env.action_size + 3)
    )
    print(
        f"Replay arrays: approximately {replay_bytes / 2**30:.2f} GiB "
        "(excludes physics/networks/compiler).",
        flush=True,
    )
    run = None
    if log_to_wandb:
        import wandb

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=metadata,
            dir=str(run_dir),
        )
    try:
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log_file:

            def progress(num_steps, metrics):
                values = {key: float(value) for key, value in metrics.items()}
                values["training/env_steps"] = int(num_steps)
                values["experiment/walltime"] = time.monotonic() - started
                if not all(math.isfinite(value) for value in values.values()):
                    raise FloatingPointError(
                        "Non-finite SAC training/evaluation metrics"
                    )
                log_file.write(json.dumps(values) + "\n")
                log_file.flush()
                print(
                    f"Step {num_steps:,}: reward={values['eval/episode_reward']:.3f}",
                    flush=True,
                )
                if run is not None:
                    run.log(values, step=int(num_steps))

            _, params, _ = sac.train(
                environment=env,
                num_envs=num_envs,
                episode_length=env._config.max_episode_steps,
                progress_fn=progress,
                network_factory=network_factory,
                wrap_env_fn=wrap_for_training,
                checkpoint_logdir=str(run_dir / "checkpoints"),
                **sac_params,
            )
        if save_policy:
            with (run_dir / "playground_params.pickle").open("wb") as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Elapsed: {time.monotonic() - started:.1f}s", flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    validate(p, args)
    main(**vars(args))
