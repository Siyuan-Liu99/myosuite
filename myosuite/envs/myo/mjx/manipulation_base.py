"""Functional reset/model randomization shared by the MJX hand tasks.

CPU MyoSuite mutates MjModel on reset. Here only small sampled parameters live
in State.info; a model is reconstructed from them without mutating the env.
"""

from abc import abstractmethod

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground import State
from mujoco_playground._src import mjx_env

from myosuite.envs.myo.mjx.mjx_base_env import MjxMyoBase, make_data
from myosuite.envs.myo.mjx.numerical_safety import sanitize_state


def uniform(rng, bounds, shape=()):
    return jax.random.uniform(
        rng, shape, minval=jp.asarray(bounds[0]), maxval=jp.asarray(bounds[1])
    )


def replace_geom_sizes(model, ids, sizes):
    """Update collision bounds as well as sizes for per-episode randomization.

    A conservative bounding sphere/AABB works for all primitives used here.
    Keeping the compiled bounds would incorrectly cull enlarged objects in Warp.
    """
    radius = jp.sum(jp.abs(sizes), axis=-1)
    aabb = jp.stack(
        (jp.zeros_like(sizes), jp.broadcast_to(radius[..., None], sizes.shape)),
        axis=-2,
    )
    # MJX-JAX retains MuJoCo's (..., 6) layout; MJX-Warp uses (..., 2, 3).
    aabb = aabb.reshape(model.geom_aabb[ids].shape)
    return model.replace(
        geom_size=model.geom_size.at[ids].set(sizes),
        geom_rbound=model.geom_rbound.at[ids].set(radius),
        geom_aabb=model.geom_aabb.at[ids].set(aabb),
    )


class MjxHandManipulationBase(MjxMyoBase):
    # Brax must reset data AND info, including goals and physical parameters.
    requires_full_reset = True

    def preprocess_spec(self, spec):
        spec = super().preprocess_spec(spec)
        if self.impl == "jax":
            # MJX-JAX has no box/ellipsoid collision kernel. The hand has a
            # small ellipsoid thumb pad; approximate that pad by a sphere.
            # Object geometries are left intact. Warp retains the original pad.
            for geom in spec.geoms:
                if (
                    geom.type != mujoco.mjtGeom.mjGEOM_ELLIPSOID
                    or not (geom.contype or geom.conaffinity)
                    or geom.name in ("keyhead", "obj", "target")
                ):
                    continue
                radius = float(max(geom.size))
                geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                geom.size = [radius, 0, 0]
        return spec

    def _model_from_info(self, info):
        return self.mjx_model

    def _get_data(self, qpos, qvel):
        capacity = self._config.contacts_per_env * self._config.num_envs
        return make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mj_model.nu),
            impl=self.impl,
            naconmax=capacity,
            naccdmax=capacity,
            njmax=self.mj_model.njmax if self.mj_model.njmax != -1 else 1000,
        )

    def _reset_state(self, rng, qpos, task_info):
        info = {
            **task_info,
            "rng": rng,
            "step_count": jp.array(0, dtype=jp.int32),
            "success_seen": jp.array(False),
        }
        data = self._get_data(qpos, jp.zeros(self.mj_model.nv))
        # Unlike make_data alone, forward populates site/geom positions and
        # proprioception before the first observation is constructed.
        data = mjx.forward(self._model_from_info(info), data)
        metrics = {
            f"{key}_reward": jp.array(0.0) for key in self._config.reward_weights
        }
        metrics.update(
            solved_frac=jp.array(0.0),
            solved_per_step=jp.array(0.0),
            success=jp.array(0.0),
        )
        return sanitize_state(
            State(
                data,
                self._get_obs(data, info),
                jp.array(0.0),
                jp.array(0.0),
                metrics,
                info,
            )
        )

    def _prepare_step(self, state):
        return state

    def _step_simulation(self, state, action):
        action = self.norm_actions(action) if self._config.norm_actions else action
        model = self._model_from_info(state.info)
        data = mjx_env.step(model, state.data, action, self._n_substeps)
        # mjx.step integrates qpos after computing derived quantities. CPU
        # MyoSuite calls forward before observing/rewarding; do the same here.
        data = mjx.forward(model, data)
        return state.replace(
            data=data,
            info={**state.info, "step_count": state.info["step_count"] + 1},
        )

    def step(self, state, action):
        state = self._step_simulation(self._prepare_step(state), action)
        terms = self._reward_terms(state.data, state.info)
        solved = terms["solved"].astype(jp.float32)
        metrics = {
            **state.metrics,
            **{f"{key}_reward": terms[key] for key in self._config.reward_weights},
            "solved_frac": solved / self._config.max_episode_steps,
            "solved_per_step": solved,
            # Emit one pulse on the FIRST success. Brax sums metrics, so its
            # eval/episode_success is an episode success rate, not a step count.
            "success": solved * (~state.info["success_seen"]),
        }
        reward = sum(
            weight * terms[key] for key, weight in self._config.reward_weights.items()
        )
        return sanitize_state(
            state.replace(
                obs=self._get_obs(state.data, state.info),
                reward=reward,
                done=terms["done"].astype(jp.float32),
                metrics=metrics,
                info={
                    **state.info,
                    "success_seen": state.info["success_seen"] | terms["solved"],
                },
            )
        )

    def _get_rewards(self, data, info):
        terms = self._reward_terms(data, info)
        return {
            key: weight * terms[key]
            for key, weight in self._config.reward_weights.items()
        }

    def _activation_cost(self, data):
        return jp.linalg.norm(data.act) / max(self.mj_model.na, 1)

    @abstractmethod
    def _reward_terms(self, data, info):
        """Raw reward components, plus scalar boolean solved and done."""
