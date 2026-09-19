"""MJX collection adapter preserving terminal observations and full resets."""

import copy

from flax import struct
import jax
import jax.numpy as jp

from myosuite.envs.myo.mjx import make
from myosuite.envs.myo.mjx.fast_sac.replay import Transition


def observation(state):
    if isinstance(state.obs, dict):
        return state.obs["state"]
    return state.obs


@struct.dataclass
class CollectionState:
    env_state: object
    rng: jax.Array
    lengths: jax.Array
    returns: jax.Array
    solved_steps: jax.Array


class VectorEnv:
    """One control step per transition; the environment owns physics substeps.

    This adapter intentionally bypasses Brax's auto-reset wrapper so replay
    always sees the terminal observation before replacing the episode state.
    Episode length is tracked independently of legacy env.info['step_count'].
    """

    def __init__(self, env_name, impl, num_envs):
        self.env = make(env_name, config_overrides={"impl": impl, "num_envs": num_envs})
        if self.env is None:
            raise ValueError(f"Unknown MJX environment: {env_name}")
        # Legacy Pose/Reach/Pen factories reuse configs. Freeze this instance
        # before constructing an eval env with a different contact capacity.
        self.env.unwrapped._config = copy.deepcopy(self.env.unwrapped._config)
        self.horizon = int(self.env._config.max_episode_steps)
        self.num_envs = num_envs
        self.action_size = self.env.action_size
        self._reset = jax.vmap(self.env.reset)
        self._step = jax.vmap(self.env.step)

    def reset(self, rng):
        rng, reset_rng = jax.random.split(rng)
        state = self._reset(jax.random.split(reset_rng, self.num_envs))
        return CollectionState(
            env_state=state,
            rng=rng,
            lengths=jp.zeros(self.num_envs, dtype=jp.int32),
            returns=jp.zeros(self.num_envs),
            solved_steps=jp.zeros(self.num_envs),
        )

    def step(self, state, action):
        stepped = self._step(state.env_state, action)
        terminated = stepped.done.astype(bool)
        lengths = state.lengths + 1
        truncated = (lengths >= self.horizon) & ~terminated
        ended = terminated | truncated
        # Both new manipulation tasks and legacy tasks expose solved_frac.
        solved = (stepped.metrics["solved_frac"] > 0).astype(jp.float32)
        returns = state.returns + stepped.reward
        solved_steps = state.solved_steps + solved
        transition = Transition(
            obs=observation(state.env_state),
            action=action,
            reward=stepped.reward,
            next_obs=observation(stepped),
            terminated=terminated,
            truncated=truncated,
        )
        rng, reset_rng = jax.random.split(state.rng)

        def reset_finished(current):
            candidate = self._reset(jax.random.split(reset_rng, self.num_envs))

            def select(fresh, old):
                mask = ended.reshape((self.num_envs,) + (1,) * (old.ndim - 1))
                return jp.where(mask, fresh, old)

            return jax.tree.map(select, candidate, current)

        # A batch-level conditional avoids generating resets when nobody ends.
        # When any environment ends, generate a full batch and select complete
        # per-env pytrees (data, obs, info, metrics), including random targets.
        next_env_state = jax.lax.cond(
            jp.any(ended), reset_finished, lambda x: x, stepped
        )
        episode = {
            "count": jp.sum(ended),
            "reward": jp.sum(jp.where(ended, returns, 0.0)),
            "length": jp.sum(jp.where(ended, lengths, 0)),
            "solved_frac": jp.sum(jp.where(ended, solved_steps / self.horizon, 0.0)),
            "solved_per_step": jp.sum(jp.where(ended, solved_steps / lengths, 0.0)),
            "success": jp.sum(ended & (solved_steps > 0)),
        }
        return (
            state.replace(
                env_state=next_env_state,
                rng=rng,
                lengths=jp.where(ended, 0, lengths),
                returns=jp.where(ended, 0.0, returns),
                solved_steps=jp.where(ended, 0.0, solved_steps),
            ),
            transition,
            episode,
        )

    def evaluate(self, policy, rng):
        """Evaluate exactly one fresh episode per world, with no auto reset."""
        state = self.reset(rng).env_state
        alive = jp.ones(self.num_envs, dtype=bool)
        zero = jp.zeros(self.num_envs)

        def step(carry, _):
            state, alive, returns, lengths, solved_steps = carry
            stepped = self._step(state, policy(observation(state)))
            # Freeze finished worlds so dropped objects do not continue falling
            # for the remaining evaluation horizon. Their extra step results
            # are discarded; unfinished worlds still advance in one GPU batch.
            state = jax.tree.map(
                lambda new, old: jp.where(
                    alive.reshape((self.num_envs,) + (1,) * (old.ndim - 1)), new, old
                ),
                stepped,
                state,
            )
            returns += jp.where(alive, state.reward, 0.0)
            lengths += alive.astype(jp.float32)
            solved_steps += (alive & (state.metrics["solved_frac"] > 0)).astype(
                jp.float32
            )
            alive &= ~state.done.astype(bool)
            return (state, alive, returns, lengths, solved_steps), None

        (_, _, returns, lengths, solved_steps), _ = jax.lax.scan(
            step, (state, alive, zero, zero, zero), None, length=self.horizon
        )
        return {
            "eval/episode_reward": jp.mean(returns),
            "eval/episode_length": jp.mean(lengths),
            "eval/episode_solved_frac": jp.mean(solved_steps / self.horizon),
            "eval/episode_solved_per_step": jp.mean(
                solved_steps / jp.maximum(lengths, 1)
            ),
            "eval/episode_success": jp.mean((solved_steps > 0).astype(jp.float32)),
        }
