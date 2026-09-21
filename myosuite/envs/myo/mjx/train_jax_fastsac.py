"""Train the JAX adaptation of Holosoma FastSAC on MyoSuite MJX tasks.

Argument parsing and --check_dependencies use only the standard library and
do not import JAX/MuJoCo or initialize a GPU. Training starts only in main().
"""

import argparse
from importlib import metadata
import math
import os
from pathlib import Path
import sys


def dependency_versions():
    result = {}
    for name in (
        "numpy",
        "jax",
        "jaxlib",
        "jax-cuda12-plugin",
        "mujoco",
        "mujoco-mjx",
        "warp-lang",
        "brax",
        "playground",
        "flax",
        "optax",
        "wandb",
    ):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    def add(name, **kwargs):
        flags = ["--" + name]
        if "_" in name:
            flags.append("--" + name.replace("_", "-"))
        p.add_argument(*flags, **kwargs)

    add(
        "env_name",
        default="MjxHandKeyTurnFixed-v0",
        help="MJX task name (CPU aliases accepted for the eight hand tasks)",
    )
    add("impl", choices=("jax", "warp"), default="warp", help="Physics backend")
    add(
        "platform",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Require this JAX device platform; no silent CPU fallback",
    )
    add("device", type=int, default=0, help="Device index within CUDA_VISIBLE_DEVICES")
    add(
        "warp_graph_cache_size",
        type=int,
        default=1,
        help="Maximum address-keyed CUDA graphs per MJX-Warp FFI callable",
    )
    add("seed", type=int, default=42)
    add("num_envs", type=int, default=64)
    add(
        "num_timesteps",
        type=int,
        default=100_000_000,
        help="Total control transitions across all environments, rounded up to a vector step",
    )
    add(
        "buffer_size",
        type=int,
        default=4096,
        help="Replay capacity in control steps PER environment",
    )
    add(
        "batch_size",
        type=int,
        default=256,
        help="Total transitions per gradient update, independent of num_envs",
    )
    add(
        "learning_starts",
        type=int,
        default=128,
        help="Collect this many vector steps before learning",
    )
    add("num_updates", type=int, default=8, help="Critic updates per vector step")
    add(
        "policy_frequency",
        type=int,
        default=4,
        help="One actor update per this many critic updates",
    )
    add(
        "num_steps",
        type=int,
        default=1,
        help="n-step replay return length; not MuJoCo substeps",
    )
    add("gamma", type=float, default=0.97)
    add("tau", type=float, default=0.125)
    add("num_atoms", type=int, default=101)
    add("num_q_networks", type=int, default=2)
    add("v_min", type=float, default=-20.0)
    add("v_max", type=float, default=20.0)
    add(
        "reward_scale",
        type=float,
        default=None,
        help="Learner-only scaling: auto=0.01 for KeyTurn/Reorient/PenTwirl, 1 otherwise",
    )
    add("actor_hidden_dim", type=int, default=512)
    add("critic_hidden_dim", type=int, default=768)
    add("actor_learning_rate", type=float, default=3e-4)
    add("critic_learning_rate", type=float, default=3e-4)
    add("alpha_learning_rate", type=float, default=3e-4)
    add("alpha_init", type=float, default=0.001)
    add("target_entropy_ratio", type=float, default=0.0)
    add("log_std_min", type=float, default=-5.0)
    add("log_std_max", type=float, default=0.0)
    add("weight_decay", type=float, default=0.001)
    add(
        "max_grad_norm", type=float, default=0.0, help="Zero disables gradient clipping"
    )
    add("autotune", action=argparse.BooleanOptionalAction, default=True)
    add("obs_normalization", action=argparse.BooleanOptionalAction, default=True)
    add("layer_norm", action=argparse.BooleanOptionalAction, default=True)
    add(
        "num_evals",
        type=int,
        default=21,
        help="Deterministic evaluations including step zero when >=2; zero disables",
    )
    add("num_eval_envs", type=int, default=64)
    add("log_interval", type=int, default=100, help="Logging interval in vector steps")
    add(
        "save_interval",
        type=int,
        default=78125,
        help="Checkpoint interval in vector steps; zero saves only at the end",
    )
    add(
        "log_dir",
        default="runs/fastsac",
        help="A unique run subdirectory is created here",
    )
    add(
        "load_checkpoint",
        default=None,
        help="Warm-start model/optimizers/normalizer; starts a fresh replay buffer and environment",
    )
    add("log_to_wandb", action="store_true")
    add("wandb_project", default="myosuite")
    add("wandb_entity", default=None)
    add(
        "check_dependencies",
        action="store_true",
        help="Print package metadata without importing any simulation or training package",
    )
    return p


def validate(p, args):
    positive = (
        "warp_graph_cache_size",
        "num_envs",
        "num_timesteps",
        "buffer_size",
        "batch_size",
        "learning_starts",
        "num_updates",
        "policy_frequency",
        "num_steps",
        "num_q_networks",
        "num_eval_envs",
        "log_interval",
    )
    for key in positive:
        if getattr(args, key) < 1:
            p.error(f"--{key} must be positive")
    for key, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            p.error(f"--{key} must be finite")
    if args.buffer_size < args.num_steps or args.learning_starts < args.num_steps:
        p.error("buffer_size and learning_starts must both be >= num_steps")
    if math.ceil(args.num_timesteps / args.num_envs) <= args.learning_starts:
        p.error(
            "num_timesteps must leave at least one vector step after learning_starts"
        )
    if not 0 < args.gamma <= 1 or not 0 < args.tau <= 1:
        p.error("gamma and tau must be in (0, 1]")
    if args.num_atoms < 2 or args.v_min >= args.v_max:
        p.error("num_atoms must be >= 2 and v_min must be < v_max")
    if min(args.actor_hidden_dim, args.critic_hidden_dim) < 4:
        p.error("network hidden dimensions must be >= 4")
    if args.log_std_min >= args.log_std_max:
        p.error("log_std_min must be < log_std_max")
    if (
        min(
            args.actor_learning_rate,
            args.critic_learning_rate,
            args.alpha_learning_rate,
            args.alpha_init,
        )
        <= 0
    ):
        p.error("learning rates and alpha_init must be positive")
    if (
        min(
            args.num_evals,
            args.save_interval,
            args.device,
            args.max_grad_norm,
            args.weight_decay,
            args.target_entropy_ratio,
        )
        < 0
    ):
        p.error(
            "intervals, device, clipping, decay and entropy ratio must be nonnegative"
        )
    if args.platform == "cpu" and args.impl == "warp":
        p.error("--impl=warp requires --platform=gpu")
    if not 0 <= args.seed < 2**32 - 1:
        p.error("seed must be between 0 and 2**32 - 2")
    if args.reward_scale is None:
        name = args.env_name.lower()
        args.reward_scale = (
            0.01 if any(s in name for s in ("keyturn", "reorient", "pentwirl")) else 1.0
        )
    if args.reward_scale <= 0:
        p.error("reward_scale must be positive")


def main():
    p = parser()
    args = p.parse_args()
    versions = dependency_versions()
    if args.check_dependencies:
        print(f"Python: {sys.version.split()[0]} ({sys.executable})")
        for name, version in versions.items():
            print(f"{name}: {version or 'NOT INSTALLED'}")
        optional = {"jax-cuda12-plugin", "wandb"}
        missing = [
            name
            for name, version in versions.items()
            if version is None and name not in optional
        ]
        print(
            "FastSAC uses the existing MJX + Flax + Optax stack; Holosoma/PyTorch are not required."
        )
        print(
            "Metadata inspection only: package presence does not validate CUDA or runtime compatibility."
        )
        if missing:
            p.exit(1, f"Missing packages: {', '.join(missing)}\n")
        return
    validate(p, args)
    # Set before any JAX import; users may override this in their shell.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Also support execution by absolute script path without an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from myosuite.envs.myo.mjx.fast_sac.train import train

    train(args, versions)


if __name__ == "__main__":
    main()
