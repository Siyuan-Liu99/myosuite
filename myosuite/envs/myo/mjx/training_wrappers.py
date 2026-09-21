"""Brax adapters for dict observations and per-episode task randomization."""

import jax
import jax.numpy as jp
from mujoco_playground import wrapper


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
        return state.replace(obs=state.obs["state"])

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

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(
            info={**state.info, self._active_key: jp.ones_like(state.done, dtype=bool)}
        )

    def step(self, state, action):
        active = state.info[self._active_key]
        # Do not expose our bookkeeping to the inner full-reset wrapper: its
        # candidate reset state has a different info structure otherwise.
        info = {
            key: value for key, value in state.info.items() if key != self._active_key
        }
        stepped = self.env.step(state.replace(info=info), action)

        def first_episode(value):
            mask = active.reshape(active.shape + (1,) * (value.ndim - active.ndim))
            return jp.where(mask, value, jp.zeros_like(value))

        return stepped.replace(
            reward=first_episode(stepped.reward),
            metrics=jax.tree.map(first_episode, stepped.metrics),
            info={
                **stepped.info,
                self._active_key: active & ~stepped.done.astype(bool),
            },
        )


def wrap_for_training(env, **kwargs):
    # A cached data/observation reset would leave new-task goals, model
    # randomization, trajectory phase and success history from the last episode.
    kwargs["full_reset"] = getattr(env.unwrapped, "requires_full_reset", False)
    wrapped = wrapper.wrap_for_brax_training(env, **kwargs)
    if getattr(env, "_evaluation_only", False):
        wrapped = FirstEpisodeMetricsWrapper(wrapped)
    return wrapped
