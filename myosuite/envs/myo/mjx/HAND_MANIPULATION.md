# MyoSuite 手部任务：MJX / Brax

本次新增环境实现及训练入口连接，**尚未执行环境测试、JIT 编译、仿真或训练**。
以下是后续 GPU 空闲时使用的命令，不代表已验证能完成训练。

## 环境名称

| CPU 环境 | MJX 环境 | episode 最大控制步数 | 控制周期 |
| --- | --- | ---: | ---: |
| `myoHandKeyTurnFixed-v0` | `MjxHandKeyTurnFixed-v0` | 200 | 0.020 s |
| `myoHandKeyTurnRandom-v0` | `MjxHandKeyTurnRandom-v0` | 200 | 0.020 s |
| `myoHandReorient8-v0` | `MjxHandReorient8-v0` | 50 | 0.010 s |
| `myoHandReorient100-v0` | `MjxHandReorient100-v0` | 50 | 0.010 s |
| `myoChallengeDieReorientP1-v0` | `MjxChallengeDieReorientP1-v0` | 150 | 0.010 s |
| `myoChallengeDieReorientP2-v0` | `MjxChallengeDieReorientP2-v0` | 150 | 0.010 s |
| `myoChallengeBaodingP1-v1` | `MjxChallengeBaodingP1-v1` | 200 | 0.025 s |
| `myoChallengeBaodingP2-v1` | `MjxChallengeBaodingP2-v1` | 200 | 0.025 s |

`mjx.make()` 及这两个训练脚本也接受表中 CPU 名称作为别名；返回的仍然是
MJX 环境。没有修改 CPU Gym 的注册。新环境默认配置在 `manipulation_config.py`。

## 使用哪个 Python

使用 **myosuite 仓库自己的 `.venv`**，而不是 `PIH-gpu/.venv`。
这次实现没有安装或更新依赖。当前本地 myosuite 环境的 Brax 版本为 0.14.1。

```bash
cd /home/lsy/pycode/myosuite/myosuite/envs/myo/mjx
source /home/lsy/pycode/myosuite/.venv/bin/activate
```

所有命令使用 JAX/Brax 训练。`--impl=warp` 选择 MJX 的 Warp 物理后端，
沿用现有转笔任务的使用方式；改为 `--impl=jax` 则选择 XLA 物理后端。
后者的手部接触几何近似见下文。可在命令前添加 `CUDA_VISIBLE_DEVICES=0`
来指定后续使用的 GPU。这里没有执行任何命令。

## PPO：八个任务的命令

按需选择一条运行。`--num_envs=1024` 是示例并行数，可按显存调整；
SAR 的四形状集合会增加接触计算，未做性能或显存测量。

```bash
python train_jax_ppo.py --env_name=MjxHandKeyTurnFixed-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_ppo.py --env_name=MjxHandKeyTurnRandom-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_ppo.py --env_name=MjxHandReorient8-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_ppo.py --env_name=MjxHandReorient100-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_ppo.py --env_name=MjxChallengeDieReorientP1-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_ppo.py --env_name=MjxChallengeDieReorientP2-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_ppo.py --env_name=MjxChallengeBaodingP1-v1 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_ppo.py --env_name=MjxChallengeBaodingP2-v1 --impl=warp --num_envs=1024 --log_to_wandb
```

## SAC：八个任务的命令

```bash
python train_jax_sac.py --env_name=MjxHandKeyTurnFixed-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_sac.py --env_name=MjxHandKeyTurnRandom-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_sac.py --env_name=MjxHandReorient8-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_sac.py --env_name=MjxHandReorient100-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_sac.py --env_name=MjxChallengeDieReorientP1-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_sac.py --env_name=MjxChallengeDieReorientP2-v0 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_sac.py --env_name=MjxChallengeBaodingP1-v1 --impl=warp --num_envs=1024 --log_to_wandb
python train_jax_sac.py --env_name=MjxChallengeBaodingP2-v1 --impl=warp --num_envs=1024 --log_to_wandb
```

PPO 使用环境的 `{"state": observation}`，SAC 入口通过
`FlatStateObservationWrapper` 取出数组，以匹配本地 Brax SAC 接口；
SAC 网络配置使用 `hidden_layer_sizes`。超参数在 `rl_cfg.py`：当前 PPO
总步数为 10M，SAC 为 100M，默认各做 16 次评估，每次 128 个 episode。

两类训练均写入 W&B **`myosuite` project**；不传 `--log_to_wandb` 则关闭。
Run 名称沿用 `环境名-MMDD-HHMM`。可以加 `--save_policy` 在训练结束时保存
`playground_params.pickle`，该文件保存在当前目录，再次保存会覆盖。

## 任务逻辑与 CPU 的对应

- **KeyTurn**：复用原模型、手部全开初态、两指到钥匙的接近距离、角度奖励、
  激活惩罚及距离失败条件。Fixed 成功阈值为 `3.14`，Random 为 `2*pi`；
  Random 的初始角度为 `[-pi/2, pi/2]`，钥匙位置每轴随机偏移 ±1 cm。
- **Reorient8/100**：使用 CPU 源码中的完整尺寸表（4 类形状 × 2/25 个尺寸），
  每个 episode 分别采样类型、尺寸及目标方向。保留 1.2 kg 物体质量、
  位置/方向/激活/掉落/bonus 奖励及肌肉长度、速度、力观测。成功条件为
  方向余弦 `>0.95` 且未掉落，位置偏离 `>0.075 m` 判失败。
- **DieReorient P1/P2**：保留 CPU 的 Euler 角相减误差及奖励
  `-100*pos_dist - rot_dist`。P1 目标位置 ±1 cm、目标 Euler 分量 ±1.57；
  P2 扩展到 ±2 cm、±3.14，并随机化尺寸 ±7 mm、质量 50–250 g 和摩擦。
  成功需位置误差 `<0.025 m`、角度误差 `<0.262`，距离 `>0.2 m` 判失败。
  保留 CPU v0 的 `hand_qpos_noMD5` 观测切片，未擅自修正其历史缺失的末端关节。
- **Baoding P1/P2**：保留相对于手掌的双球椭圆目标轨迹、双球距离奖励和
  世界高度 `<1.25 m` 的掉落条件；双球均距目标 `<0.015 m` 判成功。
  P1 固定逆时针、周期 5 秒；P2 每局随机 HOLD/CW/CCW、初始相位、
  周期 4–6 秒、椭圆半径、球半径 18–24 mm、质量 30–300 g 及摩擦。

## 与 CPU / 现有转笔移植的差异

1. **并行状态与重置**：目标、物体参数、轨迹相位及成功历史都在 `State.info`
   中；不在 `step/reset` 中修改共享的 `self.mjx_model`。
   两个训练入口为新环境启用 Playground `full_reset=True`，每局重新采样，
   同步重置 data、obs 和 info。直接使用 Brax 时也应传入
   `training_wrappers.wrap_for_training`。旧转笔等环境保留原重置方式。
   Playground 当前的 full-reset wrapper 会在每个控制步生成候选重置状态，
   之后按 done 选择，因此有额外开销，本次尚未做优化或性能测试。
2. **几何类型静态化**：MJX 的 `geom_type` 是编译期元数据。SAR 模型预置
   capsule/ellipsoid/cylinder/box 四种形状，运行时用 contact gap 让非选中
   形状不产生约束，并隐藏它们；有效形状的尺寸及碰撞边界随 episode 更新。
   原始物体 geom 仅用于保持编译时的惯性来源，关闭碰撞和显示。
   目标物体仅作可视化，关闭碰撞；配色统一为黄色，未移植 CPU 的随机颜色。
3. **物理参数**：质量变化按 CPU 源码的方式处理，不重新按新质量/尺寸计算
   转动惯量；更新随机几何的包围球/AABB，防止 Warp 用旧尺寸错误剔除接触。
4. **后端差异**：沿用转笔基类的 solver 预处理。纯 `--impl=jax` 对
   box/ellipsoid 接触没有对应 kernel，因此将手部可碰撞 ellipsoid 垫片
   近似为包围球；任务物体形状保持原样。`--impl=warp` 保留原手部几何。
   不应把两个后端的接触轨迹视为数值等价。
5. **时间与观测**：新任务按 CPU XML timestep × frame_skip 设置控制周期；
   速度观测乘控制周期 `dt`，对应 CPU，而非直接照搬转笔中的仿真 timestep。
   重置后调用 MJX forward 再构造观测，避免第一次读取未计算的位姿。
   Baoding 在 reset 时立即设置第 0 帧目标，避免沿用上一局的末帧目标。
6. **可视化**：训练使用的是从 `State.info` 重建的动态 model。现有通用
   `visu_mjx_env.py`/静态 model 播放器尚未适配逐帧恢复这些随机模型参数，
   后续播放时需同时恢复模型参数，不能仅凭 qpos 还原随机物体和目标。

## W&B 成功指标（仅本次新增环境）

| 指标 | 含义 |
| --- | --- |
| `eval/episode_solved_frac` | 各 episode 的成功步数 / 最大步数，再对评估 episode 取平均；与已有转笔一致 |
| `eval/episode_solved_per_step` | 各 episode 的成功步数 / 实际步数，再取平均；Brax 对 `_per_step` 指标按实际长度归一化 |
| `eval/episode_success` | 至少成功过一次的评估 episode 比例 |

`success` 在每局第一次成功时只记录一个 1，之后记录 0，因此 Brax 的
episode 求和不会把持续成功重复计数。以上都是评估批次统计，不是训练中
最近几百局的滑动平均，也不等同于 CPU DieReorient 的“成功超过 5 步”评分。

## 文件位置

- `playground_key_turn_v0.py`：KeyTurn Fixed / Random。
- `playground_reorient_sar_v0.py`、`sar_object_sizes.py`：Reorient8 / 100。
- `playground_die_reorient_v0.py`：DieReorient P1 / P2。
- `playground_baoding_v1.py`：Baoding P1 / P2。
- `manipulation_base.py`：状态、物理模型随机化、成功指标。
- `manipulation_config.py`：注册及各任务 CPU 对应参数。
- `training_wrappers.py`：完整重置和 SAC 观测适配。
