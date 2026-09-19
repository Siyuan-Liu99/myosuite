"""Brax adapters for dict observations and per-episode task randomization."""

from mujoco_playground import wrapper


class FlatStateObservationWrapper(wrapper.Wrapper):
    """Brax SAC consumes an array; our environments expose {"state": array}."""

    @property
    def observation_size(self):
        return self.env.observation_size["state"][-1]

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.replace(obs=state.obs["state"])

    def step(self, state, action):
        state = self.env.step(state.replace(obs={"state": state.obs}), action)
        return state.replace(obs=state.obs["state"])


def wrap_for_training(env, **kwargs):
    # A cached data/observation reset would leave new-task goals, model
    # randomization, trajectory phase and success history from the last episode.
    kwargs["full_reset"] = getattr(env.unwrapped, "requires_full_reset", False)
    return wrapper.wrap_for_brax_training(env, **kwargs)
