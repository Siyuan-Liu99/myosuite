"""Single-device FastSAC collection, learning, evaluation and logging."""

from datetime import datetime
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jp

from myosuite.envs.myo.mjx.fast_sac import checkpoint, replay
from myosuite.envs.myo.mjx.fast_sac.env import VectorEnv, observation
from myosuite.envs.myo.mjx.fast_sac.learner import FastSAC, Normalizer
from myosuite.envs.myo.mjx.fast_sac.networks import deterministic_action, sample_action
from myosuite.envs.myo.mjx.manipulation_config import CPU_ALIASES

UPSTREAM_COMMIT = "bccd4d7451640a2800ddc77e469d911a84f91994"


def train(config, versions):
    devices = jax.devices(config.platform)
    if config.device >= len(devices):
        raise ValueError(
            f"Device {config.device} requested but available devices are {devices}"
        )
    with jax.default_device(devices[config.device]):
        _train(config, versions, devices[config.device])


def _train(config, versions, device):
    experiment_started = time.monotonic()
    config.env_name = CPU_ALIASES.get(config.env_name, config.env_name)
    stamp = datetime.now()
    run_name = f"{config.env_name}-fastsac-{stamp:%m%d-%H%M}"
    log_dir = Path(config.log_dir).expanduser().resolve() / f"{run_name}-{stamp:%Y%S%f}"
    log_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"Device: {device}\nPhysics: {config.impl}\nRun directory: {log_dir}",
        flush=True,
    )
    print(
        "First collection/update/evaluation calls compile JAX programs; startup may take time.",
        flush=True,
    )
    env = VectorEnv(config.env_name, config.impl, config.num_envs)
    rng, reset_rng, init_rng = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    collection = jax.jit(env.reset)(reset_rng)
    obs_size = observation(collection.env_state).shape[-1]
    normalizer = Normalizer.create(obs_size)
    if config.obs_normalization:
        normalizer = normalizer.update(observation(collection.env_state))
    learner = FastSAC(config, obs_size, env.action_size)
    state = learner.init(init_rng)
    metadata = {
        "format_version": 1,
        "algorithm": "fastsac_jax",
        "holosoma_commit": UPSTREAM_COMMIT,
        "env_name": config.env_name,
        "obs_size": obs_size,
        "action_size": env.action_size,
        "episode_length": env.horizon,
        "ctrl_dt": float(env.env.dt),
        "config": vars(config).copy(),
        "dependencies": versions,
        "checkpoint_contents": "learner, optimizers, normalizer; no replay or simulator state",
    }
    (log_dir / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if config.load_checkpoint:
        state, normalizer = checkpoint.restore(
            config.load_checkpoint, state, normalizer, metadata
        )
        print(
            "Warm start loaded. Replay, episodes and transition count start fresh.",
            flush=True,
        )

    bytes_per_transition = 4 * (2 * obs_size + env.action_size + 1) + 2
    replay_bytes = config.buffer_size * config.num_envs * bytes_per_transition
    print(
        f"Replay: {config.buffer_size * config.num_envs:,} transitions, "
        f"approximately {replay_bytes / 2**30:.2f} GiB (excludes physics/networks/compiler).\n"
        f"Learner reward scale: {config.reward_scale}; Q support: [{config.v_min}, {config.v_max}]",
        flush=True,
    )
    buffer = replay.init_replay(
        config.buffer_size, config.num_envs, obs_size, env.action_size
    )
    episode_totals = {
        key: jp.array(0.0)
        for key in (
            "count",
            "reward",
            "length",
            "solved_frac",
            "solved_per_step",
            "success",
        )
    }

    def normalize(obs, normalizer):
        return normalizer.normalize(obs) if config.obs_normalization else obs

    def collect(params, normalizer, collection, buffer, rng, totals):
        rng, action_rng = jax.random.split(rng)
        obs = normalize(observation(collection.env_state), normalizer)
        actions, _, _ = sample_action(learner.actor, params, obs, action_rng)
        collection, transition, episode = env.step(collection, actions)
        buffer = replay.insert(
            buffer, transition.replace(reward=transition.reward * config.reward_scale)
        )
        if config.obs_normalization:
            normalizer = normalizer.update(observation(collection.env_state))
        totals = jax.tree.map(lambda x, y: x + y, totals, episode)
        return normalizer, collection, buffer, rng, totals

    # Donation is restricted to replay. Actor/target trees can share initial
    # buffers, so donating the complete learner state would be unsafe.
    collect = jax.jit(collect, donate_argnums=(3,))

    @jax.jit
    def update(state, buffer, normalizer, rng):
        rng, sample_rng, update_rng = jax.random.split(rng, 3)
        batch = replay.sample(
            buffer,
            sample_rng,
            config.batch_size * config.num_updates,
            config.num_steps,
            config.gamma,
        )
        batch["obs"] = normalize(batch["obs"], normalizer)
        batch["next_obs"] = normalize(batch["next_obs"], normalizer)
        batches = jax.tree.map(
            lambda x: x.reshape((config.num_updates, config.batch_size) + x.shape[1:]),
            batch,
        )
        state, metrics = learner.update_many(state, batches, update_rng)
        return state, rng, metrics

    evaluate = None
    eval_rng = jax.random.PRNGKey(config.seed + 1)
    if config.num_evals:
        eval_env = VectorEnv(config.env_name, config.impl, config.num_eval_envs)

        @jax.jit
        def evaluate(params, normalizer, key):
            def policy(obs):
                return deterministic_action(
                    learner.actor, params, normalize(obs, normalizer)
                )

            return eval_env.evaluate(policy, key)

    total_iterations = math.ceil(config.num_timesteps / config.num_envs)
    # Like Brax, count an initial evaluation when num_evals >= 2.
    # Remaining evaluations cover the transition budget, including the end.
    evals_after_init = max(config.num_evals - 1, 1)
    eval_iterations = (
        {
            math.ceil(i * total_iterations / evals_after_init)
            for i in range(1, evals_after_init + 1)
        }
        if config.num_evals
        else set()
    )
    run = None
    if config.log_to_wandb:
        import wandb

        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=metadata,
            dir=str(log_dir),
        )
    started = time.monotonic()
    last_log_time = started
    last_log_step = 0
    update_metrics = {}
    try:
        with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as log_file:
            if config.num_evals >= 2:
                eval_rng, key = jax.random.split(eval_rng)
                eval_started = time.monotonic()
                result = jax.device_get(evaluate(state.actor.params, normalizer, key))
                initial = {key: float(value) for key, value in result.items()}
                initial["eval/walltime"] = time.monotonic() - eval_started
                initial["training/env_steps"] = 0
                initial["experiment/walltime"] = time.monotonic() - experiment_started
                if not all(math.isfinite(value) for value in initial.values()):
                    raise FloatingPointError("Non-finite initial evaluation metrics")
                log_file.write(json.dumps(initial) + "\n")
                log_file.flush()
                if run is not None:
                    run.log(initial, step=0)
                print(
                    f"steps=0 eval/episode_reward={initial['eval/episode_reward']:.4f}",
                    flush=True,
                )
                last_log_time = time.monotonic()
            for iteration in range(1, total_iterations + 1):
                normalizer, collection, buffer, rng, episode_totals = collect(
                    state.actor.params,
                    normalizer,
                    collection,
                    buffer,
                    rng,
                    episode_totals,
                )
                if iteration > config.learning_starts:
                    state, rng, update_metrics = update(state, buffer, normalizer, rng)

                num_steps = iteration * config.num_envs
                should_log = (
                    iteration % config.log_interval == 0
                    or iteration == total_iterations
                )
                should_eval = iteration in eval_iterations
                if should_log or should_eval:
                    # Synchronize only at reporting boundaries; rates otherwise
                    # measure asynchronous dispatch rather than completed work.
                    host_updates, totals = jax.device_get(
                        (update_metrics, episode_totals)
                    )
                    now = time.monotonic()
                    metrics = {
                        f"training/{key}": float(value)
                        for key, value in host_updates.items()
                    }
                    metrics.update(
                        {
                            "training/env_steps": num_steps,
                            "training/vector_steps": iteration,
                            "training/gradient_steps": int(
                                jax.device_get(state.updates)
                            ),
                            "training/sps": (num_steps - last_log_step)
                            / max(now - last_log_time, 1e-8),
                            "training/walltime": now - started,
                            "training/episodes_in_window": int(totals["count"]),
                        }
                    )
                    if totals["count"]:
                        metrics.update(
                            {
                                f"training/episode_{key}": float(
                                    value / totals["count"]
                                )
                                for key, value in totals.items()
                                if key != "count"
                            }
                        )
                    episode_totals = jax.tree.map(jp.zeros_like, episode_totals)
                    if should_eval:
                        eval_rng, key = jax.random.split(eval_rng)
                        eval_started = time.monotonic()
                        result = jax.device_get(
                            evaluate(state.actor.params, normalizer, key)
                        )
                        metrics.update(
                            {key: float(value) for key, value in result.items()}
                        )
                        metrics["eval/walltime"] = time.monotonic() - eval_started
                    metrics["experiment/walltime"] = (
                        time.monotonic() - experiment_started
                    )
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError(
                            "Non-finite training/evaluation metrics; inspect simulator state and learning configuration."
                        )
                    log_file.write(json.dumps(metrics) + "\n")
                    log_file.flush()
                    if run is not None:
                        run.log(metrics, step=num_steps)
                    summary = f"steps={num_steps:,} sps={metrics['training/sps']:.0f}"
                    for key in (
                        "training/critic_loss",
                        "eval/episode_reward",
                        "eval/episode_success",
                    ):
                        if key in metrics:
                            summary += f" {key}={metrics[key]:.4f}"
                    if metrics.get("training/support_clip_fraction", 0.0) > 0.05:
                        summary += " [Q support clipping >5%: inspect v_min/v_max or reward_scale]"
                    print(summary, flush=True)
                    last_log_step = num_steps
                    # Exclude this evaluation from the next collection/update rate.
                    last_log_time = time.monotonic()

                if iteration == total_iterations or (
                    config.save_interval and iteration % config.save_interval == 0
                ):
                    path = log_dir / f"step_{num_steps:012d}.msgpack"
                    checkpoint.save(
                        path, state, normalizer, {**metadata, "env_steps": num_steps}
                    )
                    print(f"Saved: {path}", flush=True)
    finally:
        if run is not None:
            run.finish()
