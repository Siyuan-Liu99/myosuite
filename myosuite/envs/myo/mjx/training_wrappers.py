"""Brax adapters for dict observations and per-episode task randomization."""

import jax
import jax.numpy as jp
from mujoco_playground import wrapper

TRAINING_EPISODE_STATS = "_myosuite_training_episodes"


class TrainingEpisodeMetricsWrapper(wrapper.Wrapper):
    """Completed-episode statistics outside the batched auto-reset wrapper.

    Keep unfinished episodes across reporting boundaries. Only window totals
    are cleared by the trainer after logging; do not retain a lifetime float32
    reward sum and subtract it, which loses precision during long runs.
    """

    def __init__(self, env, episode_length):
        super().__init__(env)
        self._horizon = episode_length

    def reset(self, rng):
        state = self.env.reset(rng)
        zero = jp.zeros_like(state.reward)
        stats = {
            "returns": zero,
            "lengths": zero,
            "solved_steps": zero,
            "totals": {
                name: jp.array(0.0)
                for name in (
                    "count",
                    "reward",
                    "length",
                    "solved_frac",
                    "solved_per_step",
                    "success",
                    "numerical_failure",
                )
            },
        }
        return state.replace(info={**state.info, TRAINING_EPISODE_STATS: stats})

    def step(self, state, action):
        stats = state.info[TRAINING_EPISODE_STATS]
        # Full reset candidates do not contain this outer wrapper's info.
        info = {k: v for k, v in state.info.items() if k != TRAINING_EPISODE_STATS}
        stepped = self.env.step(state.replace(info=info), action)
        ended = stepped.done.astype(bool)
        returns = stats["returns"] + stepped.reward
        lengths = stats["lengths"] + 1
        solved_steps = stats["solved_steps"] + (stepped.metrics["solved_frac"] > 0)
        episode = {
            "count": jp.sum(ended).astype(jp.float32),
            "reward": jp.sum(jp.where(ended, returns, 0.0)),
            "length": jp.sum(jp.where(ended, lengths, 0.0)),
            "solved_frac": jp.sum(jp.where(ended, solved_steps / self._horizon, 0.0)),
            "solved_per_step": jp.sum(jp.where(ended, solved_steps / lengths, 0.0)),
            "success": jp.sum(ended & (solved_steps > 0)).astype(jp.float32),
            "numerical_failure": jp.sum(
                jp.where(
                    ended,
                    stepped.metrics.get("numerical_failure", jp.zeros_like(returns)),
                    0.0,
                )
            ),
        }
        stats = {
            "returns": jp.where(ended, 0.0, returns),
            "lengths": jp.where(ended, 0.0, lengths),
            "solved_steps": jp.where(ended, 0.0, solved_steps),
            "totals": jax.tree.map(lambda x, y: x + y, stats["totals"], episode),
        }
        return stepped.replace(info={**stepped.info, TRAINING_EPISODE_STATS: stats})


class FlatStateObservationWrapper(wrapper.Wrapper):
    """Brax SAC consumes an array; our environments expose {"state": array}."""

    def __init__(self, env, *, evaluation=False):
        super().__init__(env)
        self._evaluation_only = evaluation

    @property
    def observation_size(self):
        return self.env.observation_size["state"][-1]

    def reset(self, rng):
        state = self.env.reset(rng)
        # Declare these before Brax EpisodeWrapper constructs episode_metrics.
        # Adding new keys only outside that wrapper would break its step loop
        # for legacy Pen/Pose/Reach tasks.
        return state.replace(
            obs=state.obs["state"],
            metrics={
                **state.metrics,
                "success": jp.zeros_like(state.done),
                "solved_per_step": jp.zeros_like(state.done),
            },
        )

    def step(self, state, action):
        state = self.env.step(state.replace(obs={"state": state.obs}), action)
        return state.replace(obs=state.obs["state"])


class FirstEpisodeMetricsWrapper(wrapper.Wrapper):
    """Exclude later episodes before Brax applies its multiplication mask.

    Brax's EvalWrapper accumulates metric * active. NaN/Inf from an already
    finished world's subsequent simulation can contaminate the first-episode
    result even when active=0. Select zero explicitly for those worlds only;
    never hide a non-finite value within the episode being evaluated.
    """

    _active_key = "_myosuite_eval_active"
    _success_key = "_myosuite_eval_success_seen"

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(
            info={
                **state.info,
                self._active_key: jp.ones_like(state.done, dtype=bool),
                self._success_key: jp.zeros_like(state.done, dtype=bool),
            },
            metrics={
                **state.metrics,
                "success": jp.zeros_like(state.done),
                "solved_per_step": jp.zeros_like(state.done),
            },
        )

    def step(self, state, action):
        active = state.info[self._active_key]
        success_seen = state.info[self._success_key]
        # Do not expose our bookkeeping to the inner full-reset wrapper: its
        # candidate reset state has a different info structure otherwise.
        info = {
            key: value
            for key, value in state.info.items()
            if key not in (self._active_key, self._success_key)
        }
        stepped = self.env.step(state.replace(info=info), action)
        # Legacy Pen/Pose/Reach only expose solved_frac. Emit one pulse per
        # successful episode so Brax and FastSAC share the same success meaning.
        solved = stepped.metrics["solved_frac"] > 0
        metrics = {
            **stepped.metrics,
            "success": (solved & ~success_seen).astype(jp.float32),
            "solved_per_step": solved.astype(jp.float32),
        }

        def first_episode(value):
            mask = active.reshape(active.shape + (1,) * (value.ndim - active.ndim))
            return jp.where(mask, value, jp.zeros_like(value))

        return stepped.replace(
            reward=first_episode(stepped.reward),
            metrics=jax.tree.map(first_episode, metrics),
            info={
                **stepped.info,
                self._active_key: active & ~stepped.done.astype(bool),
                self._success_key: success_seen | (active & solved),
            },
        )


def wrap_for_training(env, **kwargs):
    # A cached data/observation reset would leave new-task goals, model
    # randomization, trajectory phase and success history from the last episode.
    kwargs["full_reset"] = getattr(env.unwrapped, "requires_full_reset", False)
    wrapped = wrapper.wrap_for_brax_training(env, **kwargs)
    if getattr(env, "_evaluation_only", False):
        wrapped = FirstEpisodeMetricsWrapper(wrapped)
    elif isinstance(env, FlatStateObservationWrapper):
        wrapped = TrainingEpisodeMetricsWrapper(wrapped, kwargs["episode_length"])
    return wrapped
