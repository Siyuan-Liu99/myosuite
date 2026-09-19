"""MJX counterpart of myobase.key_turn_v0.KeyTurnEnvV0."""

import jax
import jax.numpy as jp

from myosuite.envs.myo.mjx.manipulation_base import MjxHandManipulationBase, uniform


class MjxKeyTurnEnvV0(MjxHandManipulationBase):
    def __init__(self, config, config_overrides=None):
        super().__init__(config, config_overrides)
        self._key_bid = self.mj_model.body("key").id
        self._key_sid = self.mj_model.site("keyhead").id
        self._if_sid = self.mj_model.site("IFtip").id
        self._th_sid = self.mj_model.site("THtip").id

    def reset(self, rng):
        rng, angle_rng, position_rng = jax.random.split(rng, 3)
        qpos = jp.array(self.mj_model.qpos0).at[:-1].set(0.0)
        qpos = qpos.at[-1].set(uniform(angle_rng, self._config.key_init_range))
        offset = uniform(position_rng, (-0.01, 0.01), (3,))
        if not self._config.random_key_position:
            offset = jp.zeros(3)
        return self._reset_state(rng, qpos, {"key_offset": offset})

    def _model_from_info(self, info):
        model = self.mjx_model
        return model.replace(
            body_pos=model.body_pos.at[self._key_bid].add(info["key_offset"])
        )

    def _approach(self, data):
        key = data.site_xpos[self._key_sid]
        return key - data.site_xpos[self._if_sid], key - data.site_xpos[self._th_sid]

    def _get_obs(self, data, info):
        index, thumb = self._approach(data)
        return {
            "state": jp.concatenate(
                (
                    data.qpos[:-1],
                    data.qvel[:-1] * self.dt,
                    data.qpos[-1:],
                    data.qvel[-1:] * self.dt,
                    index,
                    thumb,
                    data.act,
                )
            )
        }

    def _reward_terms(self, data, info):
        index, thumb = self._approach(data)
        index_dist = jp.abs(jp.linalg.norm(index) - 0.030)
        thumb_dist = jp.abs(jp.linalg.norm(thumb) - 0.030)
        angle = data.qpos[-1]
        return {
            "key_turn": angle,
            "IFtip_approach": -index_dist,
            "THtip_approach": -thumb_dist,
            "act_reg": -self._activation_cost(data),
            "bonus": (angle > jp.pi / 2).astype(jp.float32) + (angle > jp.pi),
            "penalty": -(index_dist > 0.05).astype(jp.float32) - (thumb_dist > 0.05),
            "solved": angle > self._config.goal_th,
            "done": (index_dist > 0.1) | (thumb_dist > 0.1),
        }
