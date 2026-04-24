from typing import Dict, Optional, Union, Any
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from myosuite.envs.myo.mjx.mjx_base_env import MjxMyoBase


class MjxPenTwirlEnvV0(MjxMyoBase):
    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._obj_bid = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "Object"
        )
        self._eps_ball_sid = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "eps_ball"
        )
        self._obj_t_sid = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "object_top"
        )
        self._obj_b_sid = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "object_bottom"
        )

        # Site positions are in body-local frames, so this yields axis direction.
        obj_axis = self._mj_model.site_pos[self._obj_t_sid] - self._mj_model.site_pos[self._obj_b_sid]
        self._default_target_rot = jp.array(obj_axis / (jp.linalg.norm(obj_axis) + 1e-8))

    def _generate_target_rot(self, rng: jp.ndarray) -> jp.ndarray:
        if self._config.random_target:
            angle_range = self._config.target_euler_range
            euler_xy = jax.random.uniform(
                rng,
                (2,),
                minval=angle_range[0],
                maxval=angle_range[1],
            )
            x = euler_xy[0]
            y = euler_xy[1]
            target_rot = jp.array(
                [
                    jp.sin(y),
                    -jp.cos(y) * jp.sin(x),
                    jp.cos(y) * jp.cos(x),
                ]
            )
            return target_rot / (jp.linalg.norm(target_rot) + 1e-8)
        return self._default_target_rot

    @staticmethod
    def _cosine(a: jp.ndarray, b: jp.ndarray) -> jp.ndarray:
        denom = (jp.linalg.norm(a, axis=-1) * jp.linalg.norm(b, axis=-1)) + 1e-8
        return jp.sum(a * b, axis=-1) / denom

    def _pen_terms(self, data: mjx.Data, info: Dict[str, jp.ndarray]) -> Dict[str, jp.ndarray]:
        obj_pos = data.xpos[self._obj_bid]
        obj_des_pos = data.site_xpos[self._eps_ball_sid].ravel()

        obj_rot_vec = data.site_xpos[self._obj_t_sid] - data.site_xpos[self._obj_b_sid]
        obj_rot = obj_rot_vec / (jp.linalg.norm(obj_rot_vec) + 1e-8)

        obj_err_pos = obj_pos - obj_des_pos
        pos_align = jp.linalg.norm(obj_err_pos, axis=-1)
        rot_align = self._cosine(obj_rot, info["target_rot"])
        dropped = pos_align > 0.075

        act_mag = jp.where(
            self.mjx_model.na != 0,
            jp.linalg.norm(data.act, axis=-1) / self.mjx_model.na,
            0.0,
        )

        bonus = (
            1.0 * (rot_align > 0.9) * (pos_align < 0.075)
            + 5.0 * (rot_align > 0.95) * (pos_align < 0.075)
        )

        return {
            "obj_pos": obj_pos,
            "obj_des_pos": obj_des_pos,
            "obj_rot": obj_rot,
            "obj_err_pos": obj_err_pos,
            "obj_err_rot": obj_rot - info["target_rot"],
            "pos_align": pos_align,
            "rot_align": rot_align,
            "dropped": dropped,
            "act_mag": act_mag,
            "bonus": bonus,
        }

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos = jp.array(self._mj_model.qpos0)
        qpos = qpos.at[:-6].set(0.0)
        qpos = qpos.at[0].set(-1.5)
        qvel = jp.zeros(self.mjx_model.nv)

        target_rot = self._generate_target_rot(rng2)

        info = {
            "rng": rng,
            "target_rot": target_rot,
            "step_count": jp.array(0, dtype=jp.int32),
        }

        data = self._get_data(qpos, qvel)
        obs = self._get_obs(data, info)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_align_reward": zero,
            "rot_align_reward": zero,
            "act_reg_reward": zero,
            "drop_reward": zero,
            "bonus_reward": zero,
            "solved_frac": zero,
        }
        return State(data, obs, reward, done, metrics, info)

    def _get_rewards(self, data: mjx.Data, info: Dict[str, jp.ndarray]) -> Dict[str, jp.ndarray]:
        terms = self._pen_terms(data, info)
        return {
            "pos_align": -1.0 * terms["pos_align"] * self._config.reward_config.pos_align_weight,
            "rot_align": terms["rot_align"] * self._config.reward_config.rot_align_weight,
            "act_reg": -1.0 * terms["act_mag"] * self._config.reward_config.act_reg_weight,
            "drop": -1.0 * terms["dropped"] * self._config.reward_config.drop_weight,
            "bonus": terms["bonus"] * self._config.reward_config.bonus_weight,
        }

    def _get_done(self, state: State) -> float:
        terms = self._pen_terms(state.data, state.info)
        return 1.0 * terms["dropped"]

    def _get_metrics(self, state: State) -> dict:
        terms = self._pen_terms(state.data, state.info)
        solved = 1.0 * (terms["rot_align"] > 0.95) * (1.0 - 1.0 * terms["dropped"])

        return {
            "pos_align_reward": -1.0 * terms["pos_align"],
            "rot_align_reward": terms["rot_align"],
            "act_reg_reward": -1.0 * terms["act_mag"],
            "drop_reward": -1.0 * terms["dropped"],
            "bonus_reward": terms["bonus"],
            "solved_frac": solved / self._config.max_episode_steps,
        }

    def _get_info(self, state: State) -> dict:
        done = state.done

        # reset step counter if done or truncation
        truncation = jp.where(
            state.info["step_count"] >= self._config.max_episode_steps,
            1.0 - done,
            jp.array(0.0),
        )
        step_count = jp.where(
            jp.logical_or(done, truncation),
            jp.array(0, dtype=jp.int32),
            state.info["step_count"],
        )

        # reset target orientation if done or truncation
        rng, rng1 = jax.random.split(state.info["rng"])
        target_rot = jp.where(
            jp.logical_or(done, truncation),
            self._generate_target_rot(rng1),
            state.info["target_rot"],
        )

        info = {
                **state.info,
                "rng": rng,
                "step_count": step_count,
                "target_rot": target_rot,
            }

        return info

    def _get_obs(self, data: mjx.Data, info: Dict) -> jp.ndarray:
        """Observe hand/object kinematics and position/rotation errors."""
        terms = self._pen_terms(data, info)
        obs = jp.concatenate(
            [
                data.qpos[:-6],
                terms["obj_pos"],
                data.qvel[-6:] * self.mjx_model.opt.timestep,
                terms["obj_rot"],
                info["target_rot"],
                terms["obj_err_pos"],
                terms["obj_err_rot"],
                data.act,
            ]
        )
        return {"state": obs}
