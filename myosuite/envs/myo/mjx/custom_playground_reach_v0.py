from typing import Dict
import jax
import jax.numpy as jp
from mujoco_playground import State
from myosuite.envs.myo.mjx.playground_reach_v0 import MjxReachEnvV0


class CustomMjxReachEnvV0(MjxReachEnvV0):
    def _get_rewards(self, data, info):
        reach_err = self._reach_err(data, info)
        reach_dist = jp.linalg.norm(reach_err, axis=-1)
        
        far_th = jp.where(
            data.time > 2.0 * self.mjx_model.opt.timestep,
            self._config.far_th * self.n_targets,
            jp.inf,
        )

        reach = -1.0 * reach_dist * self._config.reward_config.reach_weight
        bonus = (
            1.0 * (reach_dist < 2 * self.near_th) + 1.0 * (reach_dist < self.near_th)
        ) * self._config.reward_config.bonus_scale
        penalty = -1.0 * (reach_dist > far_th) * self._config.reward_config.penalty_scale

        epsilon = 1e-4
        log_reach = -1.0 * (reach_dist + jp.log(reach_dist + epsilon**2)) * self._config.reward_config.log_reach_weight

        return {"reach": reach, "bonus": bonus, "penalty": penalty, "log_reach": log_reach}
    
    def _get_done(self, state: State) -> float:
        reach_err = self._reach_err(state.data, state.info)
        reach_dist = jp.linalg.norm(reach_err, axis=-1)
        far_th = jp.where(
            state.data.time > 2.0 * self.mjx_model.opt.timestep,
            self._config.far_th * self.n_targets,
            jp.inf,
        )
        done = 1.0 * (reach_dist > far_th)
        
        return done
    
    def _get_metrics(self, state: State) -> dict:
        reach_err = self._reach_err(state.data, state.info)
        reach_dist = jp.linalg.norm(reach_err, axis=-1)
        solved = 1.0 * (reach_dist < self.near_th)

        far_th = jp.where(
            state.data.time > 2.0 * self.mjx_model.opt.timestep,
            self._config.far_th * self.n_targets,
            jp.inf,
        )

        reach = -1.0 * reach_dist
        bonus = 1.0 * (reach_dist < 2 * self.near_th) + 1.0 * (reach_dist < self.near_th)
        penalty = -1.0 * (reach_dist > far_th)

        return {
            "reach_reward": reach,
            "bonus_reward": bonus,
            "penalty_reward": penalty,
            "solved_frac": solved / self._config.max_episode_steps
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

        # reset targets if done or truncation
        rng, rng1 = jax.random.split(state.info["rng"])
        targets = jp.where(
            jp.logical_or(done, truncation),
            self.generate_target_pose(rng1),
            state.info["targets"],
        )
        
        info={
                **state.info,
                "rng": rng,
                "step_count": step_count,
                "targets": targets,
            }
        
        return info
