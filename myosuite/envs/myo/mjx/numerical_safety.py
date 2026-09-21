"""Keep failed physics transitions out of observation statistics and replay.

This is an environment boundary guard, not a repair for non-finite network
parameters. Invalid episodes terminate and expose a numerical_failure metric.
"""

import jax
import jax.numpy as jp


def sanitize_state(state):
    """Sanitize one unbatched state; compatible with the environment's vmap.

    Explicit zero replacements avoid nan_to_num's default huge finite values
    for infinity, which can overflow normalization or squared critic losses.
    Keep physics data untouched: the auto-reset wrapper replaces it on done.
    """
    obs_bad = jp.any(
        jp.stack([~jp.all(jp.isfinite(value)) for value in jax.tree.leaves(state.obs)])
    )
    physics_bad = jp.any(
        jp.stack(
            [
                ~jp.all(jp.isfinite(getattr(state.data, name)))
                for name in ("qpos", "qvel", "qacc", "act")
            ]
        )
    )
    reward_bad = ~jp.isfinite(state.reward)
    metrics_bad = jp.any(
        jp.stack([~jp.all(jp.isfinite(value)) for value in state.metrics.values()])
    )
    failed = obs_bad | physics_bad | reward_bad | metrics_bad | ~jp.isfinite(state.done)

    def clean(value):
        value = jp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        # A finite component of a failed simulation can still be enormous.
        return jp.where(failed, jp.zeros_like(value), value)

    return state.replace(
        obs=jax.tree.map(clean, state.obs),
        reward=clean(state.reward),
        done=jp.where(failed, jp.ones_like(state.done), state.done),
        metrics={
            **jax.tree.map(clean, state.metrics),
            "numerical_failure": failed.astype(jp.float32),
            "nonfinite_observation": obs_bad.astype(jp.float32),
            "nonfinite_physics": physics_bad.astype(jp.float32),
            "nonfinite_reward": reward_bad.astype(jp.float32),
        },
    )
