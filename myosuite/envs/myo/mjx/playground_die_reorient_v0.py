"""MJX counterpart of myochallenge.reorient_v0.ReorientEnvV0."""

import jax
import jax.numpy as jp

from myosuite.envs.myo.mjx.manipulation_base import (
    MjxHandManipulationBase,
    replace_geom_sizes,
    uniform,
)
from myosuite.utils.quat_math_jax import euler2quat, mat2euler


class MjxDieReorientEnvV0(MjxHandManipulationBase):
    def __init__(self, config, config_overrides=None):
        super().__init__(config, config_overrides)
        self._obj_bid = self.mj_model.body("Object").id
        self._target_bid = self.mj_model.body("target").id
        self._obj_sid = self.mj_model.site("object_o").id
        self._target_sid = self.mj_model.site("target_o").id
        self._target_gid = self.mj_model.geom("target_dice").id
        start = int(self.mj_model.body_geomadr[self._obj_bid])
        count = int(self.mj_model.body_geomnum[self._obj_bid])
        self._obj_gids = jp.arange(start, start + count)
        # Both bodies are world children and both origin sites have local pos=0.
        self._goal_init_pos = self.mjx_model.body_pos[self._target_bid]
        self._goal_offset = self._goal_init_pos - self.mjx_model.body_pos[self._obj_bid]

    def reset(self, rng):
        rng, pos_rng, rot_rng, mass_rng, size_rng, friction_rng = jax.random.split(
            rng, 6
        )
        delta = jp.array(self._config.obj_friction_change)
        friction = self.mjx_model.geom_friction[self._obj_gids]
        info = {
            "goal_pos": self._goal_init_pos
            + uniform(pos_rng, self._config.goal_pos, (3,)),
            "goal_quat": euler2quat(uniform(rot_rng, self._config.goal_rot, (3,))),
            "object_mass": uniform(mass_rng, self._config.obj_mass_range),
            "size_delta": uniform(
                size_rng, (-self._config.obj_size_change, self._config.obj_size_change)
            ),
            "object_friction": uniform(
                friction_rng, (friction - delta, friction + delta), friction.shape
            ),
        }
        # Keep v0's historical omission of the last hand joint, as on CPU.
        qpos = jp.array(self.mj_model.qpos0).at[:-7].set(0.0).at[0].set(-1.5)
        return self._reset_state(rng, qpos, info)

    def _model_from_info(self, info):
        model = self.mjx_model
        delta = info["size_delta"]
        size = model.geom_size[self._obj_gids]
        # First 12 geoms are edge capsules; the last 3 are overlapping boxes.
        size = size.at[:-3, 1].add(delta).at[-3:].add(delta)
        model = replace_geom_sizes(model, self._obj_gids, size)
        target_size = model.geom_size[self._target_gid] + delta
        model = replace_geom_sizes(model, self._target_gid, target_size)
        initial_pos = self.mjx_model.geom_pos[self._obj_gids]
        geom_pos = jp.sign(initial_pos) * (jp.abs(initial_pos) + delta)
        return model.replace(
            geom_pos=model.geom_pos.at[self._obj_gids].set(geom_pos),
            geom_friction=model.geom_friction.at[self._obj_gids].set(
                info["object_friction"]
            ),
            body_mass=model.body_mass.at[self._obj_bid].set(info["object_mass"]),
            body_pos=model.body_pos.at[self._target_bid].set(info["goal_pos"]),
            body_quat=model.body_quat.at[self._target_bid].set(info["goal_quat"]),
        )

    def _pose_terms(self, data):
        obj_pos = data.site_xpos[self._obj_sid]
        goal_pos = data.site_xpos[self._target_sid]
        obj_rot = mat2euler(data.site_xmat[self._obj_sid].reshape(3, 3))
        goal_rot = mat2euler(data.site_xmat[self._target_sid].reshape(3, 3))
        # Deliberately retain CPU v0's Euler subtraction, not quaternion distance.
        return (
            obj_pos,
            goal_pos,
            goal_pos - obj_pos - self._goal_offset,
            obj_rot,
            goal_rot,
            goal_rot - obj_rot,
        )

    def _get_obs(self, data, info):
        obj_pos, goal_pos, pos_err, obj_rot, goal_rot, rot_err = self._pose_terms(data)
        return {
            "state": jp.concatenate(
                (
                    data.qpos[:-7],
                    data.qvel[:-6] * self.dt,
                    obj_pos,
                    goal_pos,
                    pos_err,
                    obj_rot,
                    goal_rot,
                    rot_err,
                    data.act,
                )
            )
        }

    def _reward_terms(self, data, info):
        _, _, pos_err, _, _, rot_err = self._pose_terms(data)
        pos_dist, rot_dist = jp.linalg.norm(pos_err), jp.linalg.norm(rot_err)
        dropped = pos_dist > self._config.drop_th
        return {
            "pos_dist": -pos_dist,
            "rot_dist": -rot_dist,
            "bonus": (pos_dist < 2 * self._config.pos_th).astype(jp.float32)
            + (pos_dist < self._config.pos_th),
            "act_reg": -self._activation_cost(data),
            "penalty": -dropped.astype(jp.float32),
            "solved": (pos_dist < self._config.pos_th)
            & (rot_dist < self._config.rot_th)
            & (~dropped),
            "done": dropped,
        }
