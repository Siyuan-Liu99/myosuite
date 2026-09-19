"""Flax checkpoints and a small inference loader (no Holosoma dependency)."""

import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jp

from myosuite.envs.myo.mjx.fast_sac.learner import Normalizer
from myosuite.envs.myo.mjx.fast_sac.networks import Actor, deterministic_action


def save(path, learner, normalizer, metadata):
    path = Path(path)
    state = jax.device_get({"learner": learner, "normalizer": normalizer})
    temporary = path.with_suffix(".msgpack.tmp")
    temporary.write_bytes(serialization.to_bytes(state))
    temporary.replace(path)
    meta_path = path.with_suffix(".json")
    temporary_meta = meta_path.with_suffix(".json.tmp")
    temporary_meta.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_meta.replace(meta_path)


def restore(path, learner, normalizer, expected):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    for key in (
        "format_version",
        "env_name",
        "obs_size",
        "action_size",
        "episode_length",
        "ctrl_dt",
    ):
        if metadata[key] != expected[key]:
            raise ValueError(
                f"Checkpoint {key} mismatch: {metadata[key]} != {expected[key]}"
            )
    for key in (
        "actor_hidden_dim",
        "critic_hidden_dim",
        "num_atoms",
        "num_q_networks",
        "layer_norm",
        "log_std_min",
        "log_std_max",
        "obs_normalization",
        "v_min",
        "v_max",
        "reward_scale",
        "gamma",
    ):
        if metadata["config"][key] != expected["config"][key]:
            raise ValueError(f"Checkpoint configuration mismatch: {key}")
    result = serialization.from_bytes(
        {"learner": learner, "normalizer": normalizer}, path.read_bytes()
    )
    result = jax.device_put(result)
    return result["learner"], result["normalizer"]


def load_policy(path):
    """Return (policy, metadata). Policy takes raw obs[..., obs_size].

    Output is the same normalized [-1, 1] action expected by env.step. Do not
    apply a second muscle-action mapping. Parameters are Flax, not .pt/ONNX.
    """
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported FastSAC checkpoint format")
    config = metadata["config"]
    state = serialization.msgpack_restore(path.read_bytes())
    actor = Actor(
        metadata["action_size"],
        config["actor_hidden_dim"],
        config["log_std_min"],
        config["log_std_max"],
        config["layer_norm"],
    )
    # Inference only needs actor/normalizer arrays on the device, not the
    # critic ensemble and optimizer buffers also present in the checkpoint.
    normalizer = Normalizer(**jax.tree.map(jp.asarray, state["normalizer"]))
    params = jax.tree.map(jp.asarray, state["learner"]["actor"]["params"])

    def policy(obs):
        if config["obs_normalization"]:
            obs = normalizer.normalize(obs)
        return deterministic_action(actor, params, obs)

    return jax.jit(policy), metadata
