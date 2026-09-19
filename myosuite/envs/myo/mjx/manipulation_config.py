"""CPU task presets for the eight MJX hand manipulation environments."""

import copy
import math

from etils import epath
from ml_collections import config_dict

from myosuite.envs.myo.mjx import myo_registry as registry
from myosuite.envs.myo.mjx.playground_baoding_v1 import MjxBaodingEnvV1
from myosuite.envs.myo.mjx.playground_die_reorient_v0 import MjxDieReorientEnvV0
from myosuite.envs.myo.mjx.playground_key_turn_v0 import MjxKeyTurnEnvV0
from myosuite.envs.myo.mjx.playground_reorient_sar_v0 import MjxSarReorientEnvV0


MANIPULATION_ENVS = (
    "MjxHandKeyTurnFixed-v0",
    "MjxHandKeyTurnRandom-v0",
    "MjxHandReorient8-v0",
    "MjxHandReorient100-v0",
    "MjxChallengeDieReorientP1-v0",
    "MjxChallengeDieReorientP2-v0",
    "MjxChallengeBaodingP1-v1",
    "MjxChallengeBaodingP2-v1",
)
CPU_ALIASES = {"myo" + name[3:]: name for name in MANIPULATION_ENVS}


def _config(base, model_name, horizon, frame_skip, sim_dt=0.002, **kwargs):
    config = copy.deepcopy(base)
    config.update(
        dict(
            model_path=epath.resource_path("myosuite")
            / "envs/myo/assets/hand"
            / model_name,
            max_episode_steps=horizon,
            sim_dt=sim_dt,
            ctrl_dt=frame_skip * sim_dt,
            contacts_per_env=256,
            **kwargs,
        )
    )
    return config


def register_hand_manipulation_tasks(base):
    """Register independent config factories; calling make never mutates presets."""
    presets = {}
    for random in (False, True):
        name = f"MjxHandKeyTurn{'Random' if random else 'Fixed'}-v0"
        presets[name] = (
            MjxKeyTurnEnvV0,
            _config(
                base,
                "myohand_keyturn.xml",
                200,
                10,
                key_init_range=(-math.pi / 2, math.pi / 2) if random else (0.0, 0.0),
                goal_th=2 * math.pi if random else 3.14,
                random_key_position=random,
                reward_weights=config_dict.create(
                    key_turn=1.0,
                    IFtip_approach=10.0,
                    THtip_approach=10.0,
                    act_reg=1.0,
                    bonus=4.0,
                    penalty=25.0,
                ),
            ),
        )
    for count in (8, 100):
        presets[f"MjxHandReorient{count}-v0"] = (
            MjxSarReorientEnvV0,
            _config(
                base,
                "myohand_sar.xml",
                50,
                5,
                num_objects=count,
                reward_weights=config_dict.create(
                    pos_align=1.0,
                    rot_align=1.0,
                    act_reg=5.0,
                    drop=5.0,
                    bonus=10.0,
                ),
            ),
        )
    for phase in (1, 2):
        p2 = phase == 2
        presets[f"MjxChallengeDieReorientP{phase}-v0"] = (
            MjxDieReorientEnvV0,
            _config(
                base,
                "myohand_die.xml",
                150,
                5,
                goal_pos=(-0.020, 0.020) if p2 else (-0.010, 0.010),
                goal_rot=(-3.14, 3.14) if p2 else (-1.57, 1.57),
                obj_size_change=0.007 if p2 else 0.0,
                obj_mass_range=(0.050, 0.250) if p2 else (0.108, 0.108),
                obj_friction_change=(0.2, 0.001, 0.00002) if p2 else (0.0, 0.0, 0.0),
                pos_th=0.025,
                rot_th=0.262,
                drop_th=0.200,
                reward_weights=config_dict.create(
                    pos_dist=100.0,
                    rot_dist=1.0,
                    bonus=0.0,
                    act_reg=0.0,
                    penalty=0.0,
                ),
            ),
        )
        presets[f"MjxChallengeBaodingP{phase}-v1"] = (
            MjxBaodingEnvV1,
            _config(
                base,
                "myohand_baoding.xml",
                200,
                10,
                sim_dt=0.0025,
                goal_time_period=(4.0, 6.0) if p2 else (5.0, 5.0),
                goal_xrange=(0.020, 0.030) if p2 else (0.025, 0.025),
                goal_yrange=(0.022, 0.032) if p2 else (0.028, 0.028),
                obj_size_range=(0.018, 0.024) if p2 else (0.022, 0.022),
                obj_mass_range=(0.030, 0.300) if p2 else (0.043, 0.043),
                obj_friction_change=(0.2, 0.001, 0.00002) if p2 else (0.0, 0.0, 0.0),
                random_task=p2,
                drop_th=1.25,
                proximity_th=0.015,
                reward_weights=config_dict.create(
                    pos_dist_1=5.0, pos_dist_2=5.0, act_reg=0.0
                ),
            ),
        )
    for name, (env_class, config) in presets.items():
        registry.register_environment(
            name, env_class, lambda config=config: copy.deepcopy(config)
        )
