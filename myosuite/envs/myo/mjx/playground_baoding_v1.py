"""MJX counterpart of myochallenge.baoding_v1.BaodingEnvV1."""

import jax
import jax.numpy as jp

from myosuite.envs.myo.mjx.manipulation_base import (
    MjxHandManipulationBase,
    replace_geom_sizes,
    uniform,
)


class MjxBaodingEnvV1(MjxHandManipulationBase):
    def __init__(self, config, config_overrides=None):
        super().__init__(config, config_overrides)
        self._ball_bids = jp.array(
            [self.mj_model.body(name).id for name in ("ball1", "ball2")]
        )
        self._ball_gids = jp.array(
            [self.mj_model.geom(name).id for name in ("ball1", "ball2")]
        )
        self._ball_sids = jp.array(
            [self.mj_model.site(name).id for name in ("ball1_site", "ball2_site")]
        )
        self._target_sids = jp.array(
            [self.mj_model.site(name).id for name in ("target1_site", "target2_site")]
        )

    def reset(self, rng):
        keys = jax.random.split(rng, 9)
        (
            rng,
            task_rng,
            angle_rng,
            period_rng,
            x_rng,
            y_rng,
            mass_rng,
            size_rng,
            friction_rng,
        ) = keys
        task = (
            jax.random.randint(task_rng, (), 0, 3)
            if self._config.random_task
            else jp.array(2, dtype=jp.int32)
        )
        start_angle = (
            uniform(angle_rng, (0.0, 2 * jp.pi))
            if self._config.random_task
            else jp.array(jp.pi / 4)
        )
        friction = self.mjx_model.geom_friction[self._ball_gids]
        delta = jp.array(self._config.obj_friction_change)
        info = {
            # CPU enum: HOLD=0, CW=1, CCW=2.
            "task": task,
            "start_angles": jp.stack((start_angle, start_angle - jp.pi)),
            "time_period": uniform(period_rng, self._config.goal_time_period),
            "x_radius": uniform(x_rng, self._config.goal_xrange),
            "y_radius": uniform(y_rng, self._config.goal_yrange),
            "ball_mass": uniform(mass_rng, self._config.obj_mass_range, (2,)),
            "ball_radius": uniform(size_rng, self._config.obj_size_range, (2,)),
            "ball_friction": uniform(
                friction_rng, (friction - delta, friction + delta), (2, 3)
            ),
            "target_phase": jp.array(0.0),
        }
        qpos = jp.array(self.mj_model.qpos0).at[:-14].set(0.0).at[0].set(-1.57)
        return self._reset_state(rng, qpos, info)

    def _prepare_step(self, state):
        # CPU updates the target BEFORE stepping, starting with counter=0.
        sign = jp.array([0.0, -1.0, 1.0])[state.info["task"]]
        phase = (
            sign
            * 2
            * jp.pi
            * state.info["step_count"]
            * self.dt
            / state.info["time_period"]
        )
        return state.replace(info={**state.info, "target_phase": phase})

    def _model_from_info(self, info):
        model = self.mjx_model
        sizes = model.geom_size[self._ball_gids].at[:, 0].set(info["ball_radius"])
        model = replace_geom_sizes(model, self._ball_gids, sizes)
        angles = info["start_angles"] + info["target_phase"]
        target_pos = model.site_pos[self._target_sids]
        target_pos = target_pos.at[:, 0].set(info["x_radius"] * jp.cos(angles) - 0.0125)
        target_pos = target_pos.at[:, 1].set(info["y_radius"] * jp.sin(angles) - 0.07)
        # Target sites are attached to the palm. Changing model.site_pos keeps
        # the trajectory in the moving palm frame, not in fixed world axes.
        return model.replace(
            site_pos=model.site_pos.at[self._target_sids].set(target_pos),
            body_mass=model.body_mass.at[self._ball_bids].set(info["ball_mass"]),
            geom_friction=model.geom_friction.at[self._ball_gids].set(
                info["ball_friction"]
            ),
        )

    def _get_obs(self, data, info):
        balls = data.site_xpos[self._ball_sids]
        targets = data.site_xpos[self._target_sids]
        return {
            "state": jp.concatenate(
                (
                    data.qpos[:-14],
                    balls[0],
                    data.qvel[-12:-9] * self.dt,
                    balls[1],
                    data.qvel[-6:-3] * self.dt,
                    targets[0],
                    targets[1],
                    targets[0] - balls[0],
                    targets[1] - balls[1],
                    data.act,
                )
            )
        }

    def _reward_terms(self, data, info):
        balls = data.site_xpos[self._ball_sids]
        targets = data.site_xpos[self._target_sids]
        distances = jp.linalg.norm(targets - balls, axis=-1)
        dropped = jp.any(balls[:, 2] < self._config.drop_th)
        return {
            "pos_dist_1": -distances[0],
            "pos_dist_2": -distances[1],
            "act_reg": -self._activation_cost(data),
            "solved": jp.all(distances < self._config.proximity_th) & (~dropped),
            "done": dropped,
        }
