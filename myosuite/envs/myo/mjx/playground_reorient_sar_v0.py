"""MJX versions of the SAR 8-object and 100-object reorientation tasks."""

import jax
import jax.numpy as jp
import mujoco

from myosuite.envs.myo.mjx.manipulation_base import (
    MjxHandManipulationBase,
    replace_geom_sizes,
    uniform,
)
from myosuite.envs.myo.mjx.sar_object_sizes import SAR_8_SIZES, SAR_100_SIZES
from myosuite.utils.quat_math_jax import euler2quat


class MjxSarReorientEnvV0(MjxHandManipulationBase):
    _SHAPES = ("capsule", "ellipsoid", "cylinder", "box")
    _TYPES = (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_BOX,
    )

    def preprocess_spec(self, spec):
        # Keep the original geometry for inertial inference, but replace its
        # collision/visual role with four shapes of STATIC types. geom_type is
        # compile-time metadata in MJX and cannot be sampled inside jax.jit.
        for name in ("obj", "target"):
            geom = spec.geom(name)
            geom.contype = 0
            geom.conaffinity = 0
            geom.rgba = [0, 0, 0, 0]
        spec = super().preprocess_spec(spec)
        condim = 4 if self._config.num_objects == 8 else 3
        for body_name, prefix in (("Object", "sar_obj"), ("target", "sar_target")):
            body = spec.body(body_name)
            for shape, geom_type in zip(self._SHAPES, self._TYPES):
                body.add_geom(
                    name=f"{prefix}_{shape}",
                    type=geom_type,
                    size=[0.015, 0.015, 0.045],
                    density=0,
                    contype=1 if body_name == "Object" else 0,
                    conaffinity=1 if body_name == "Object" else 0,
                    condim=condim,
                    rgba=[1, 0.9, 0, 0],
                )
        return spec

    def __init__(self, config, config_overrides=None):
        super().__init__(config, config_overrides)
        self._obj_bid = self.mj_model.body("Object").id
        self._target_bid = self.mj_model.body("target").id
        self._eps_sid = self.mj_model.site("eps_ball").id
        self._obj_gids = jp.array(
            [self.mj_model.geom(f"sar_obj_{shape}").id for shape in self._SHAPES]
        )
        self._target_gids = jp.array(
            [self.mj_model.geom(f"sar_target_{shape}").id for shape in self._SHAPES]
        )
        self._marker_gids = jp.array(
            [self.mj_model.geom(name).id for name in ("top", "bot", "t_top", "t_bot")]
        )
        self._top_gid = self.mj_model.geom("top").id
        self._bot_gid = self.mj_model.geom("bot").id
        self._ttop_gid = self.mj_model.geom("t_top").id
        self._tbot_gid = self.mj_model.geom("t_bot").id
        # The CPU environment keeps these INITIAL lengths after resizing.
        self._obj_length = jp.linalg.norm(
            self.mjx_model.geom_pos[self._top_gid]
            - self.mjx_model.geom_pos[self._bot_gid]
        )
        self._target_length = jp.linalg.norm(
            self.mjx_model.geom_pos[self._ttop_gid]
            - self.mjx_model.geom_pos[self._tbot_gid]
        )
        self._sizes = jp.array(
            SAR_8_SIZES if self._config.num_objects == 8 else SAR_100_SIZES
        )

    def reset(self, rng):
        rng, shape_rng, size_rng, target_rng = jax.random.split(rng, 4)
        shape = jax.random.randint(shape_rng, (), 0, 4)
        size_index = jax.random.randint(size_rng, (), 0, self._sizes.shape[1])
        euler = uniform(target_rng, ((-1.0, -0.8, 0.0), (1.0, 1.2, 0.0)), (3,))
        qpos = jp.array(self.mj_model.qpos0).at[:-6].set(0.0).at[0].set(-1.5)
        return self._reset_state(
            rng,
            qpos,
            {
                "shape_index": shape,
                "object_size": self._sizes[shape, size_index],
                "target_quat": euler2quat(euler),
            },
        )

    def _model_from_info(self, info):
        model = self.mjx_model
        size = info["object_size"]
        shape = info["shape_index"]
        model = replace_geom_sizes(model, self._obj_gids, jp.tile(size, (4, 1)))
        model = replace_geom_sizes(model, self._target_gids, jp.tile(size, (4, 1)))
        active = jp.arange(4) == shape
        # A large gap makes inactive contacts non-constraining (dist must be
        # < margin-gap). Do not move spare shapes away: a rotated spare could
        # otherwise intersect the floor. All sizes remain strictly positive.
        gap = jp.where(active, 0.0, 1e6)
        rgba = jp.tile(jp.array([1.0, 0.9, 0.0, 1.0]), (4, 1))
        rgba = rgba.at[:, 3].set(active.astype(jp.float32))
        half_axis = jp.where(
            shape == 0, 1.3 * size[1], jp.where(shape == 2, size[1], size[2])
        )
        marker_positions = (
            jp.zeros((4, 3)).at[:, 2].set(half_axis * jp.array([1.0, -1.0, 1.0, -1.0]))
        )
        return model.replace(
            geom_gap=model.geom_gap.at[self._obj_gids].set(gap),
            geom_rgba=model.geom_rgba.at[self._obj_gids]
            .set(rgba)
            .at[self._target_gids]
            .set(rgba),
            geom_pos=model.geom_pos.at[self._marker_gids].set(marker_positions),
            body_mass=model.body_mass.at[self._obj_bid].set(1.2),
            body_quat=model.body_quat.at[self._target_bid].set(info["target_quat"]),
        )

    def _orientation_terms(self, data):
        obj_rot = (
            data.geom_xpos[self._top_gid] - data.geom_xpos[self._bot_gid]
        ) / self._obj_length
        target_rot = (
            data.geom_xpos[self._ttop_gid] - data.geom_xpos[self._tbot_gid]
        ) / self._target_length
        pos_err = data.xpos[self._obj_bid] - data.site_xpos[self._eps_sid]
        return obj_rot, target_rot, pos_err

    def _get_obs(self, data, info):
        obj_rot, target_rot, pos_err = self._orientation_terms(data)
        # Match ProprioceptiveEnvV0._setup's observation order (act is appended
        # by CPU BaseV0), including all three muscle proprioception signals.
        return {
            "state": jp.concatenate(
                (
                    data.qpos[:-6],
                    data.xpos[self._obj_bid],
                    data.qvel[-6:] * self.dt,
                    obj_rot,
                    target_rot,
                    pos_err,
                    obj_rot - target_rot,
                    data.actuator_length,
                    data.actuator_velocity,
                    data.actuator_force,
                    data.act,
                )
            )
        }

    def _reward_terms(self, data, info):
        obj_rot, target_rot, pos_err = self._orientation_terms(data)
        pos_dist = jp.linalg.norm(pos_err)
        cosine = jp.dot(obj_rot, target_rot) / jp.maximum(
            jp.linalg.norm(obj_rot) * jp.linalg.norm(target_rot), 1e-8
        )
        dropped = pos_dist > 0.075
        return {
            "pos_align": -pos_dist,
            "rot_align": cosine,
            "act_reg": -self._activation_cost(data),
            "drop": -dropped.astype(jp.float32),
            "bonus": ((cosine > 0.9).astype(jp.float32) + 5.0 * (cosine > 0.95))
            * (pos_dist < 0.075),
            "solved": (cosine > 0.95) & (~dropped),
            "done": dropped,
        }
