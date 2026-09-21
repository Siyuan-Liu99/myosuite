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

## 三张 RTX 2080 Ti：SAC / FastSAC 对比队列

当前只跑 **5 个任务 × 2 种算法 × seed 42 = 10 次训练**，每卡一个进程，
每个脚本逐条执行 Python 命令，没有任务循环。脚本结束一条才执行下一条，
任何一条报错会停止该卡队列。其余任务仅保留在本文档，不进入当前队列。

| 当前任务 | 环境实现 | episode 最大控制步数 |
| --- | --- | ---: |
| `MjxHandKeyTurnRandom-v0` | `playground_key_turn_v0.py` | 200 |
| `MjxHandReorient100-v0` | `playground_reorient_sar_v0.py` | 50 |
| `MjxHandPenTwirlRandom-v0` | `playground_pen_v0.py` | 50 |
| `MjxFingerPoseRandom-v0` | `playground_pose_v0.py` | 100 |
| `MjxHandReachRandom-v0` | `playground_reach_v0.py` | 100 |

Pose 本轮选择手指随机姿态；同一个实现文件中的肘部姿态任务不加入本轮。
Pen/Pose/Reach 的 Brax 包装器使用完整 reset，保持新一局的目标、观测和物理状态一致。

### 统一预算与显存配置

| 参数 | Brax SAC | JAX FastSAC |
| --- | --- | --- |
| 物理后端 | MJX-Warp | MJX-Warp |
| 单卡环境数 | 64 | 64 |
| 总控制步数 | 100,000,000 | 100,000,000 |
| seed | 42 | 42 |
| 每次梯度更新 batch | 256 | 256 |
| 回放总容量 | 262,144 条 | 262,144 条 |
| 回放参数 | `max_replay_size=262144` | `buffer_size=4096`，每环境容量 |
| 预热采样 | `min_replay_size=8192` | `learning_starts=128` × 64 = 8192 |
| 折扣率 / 学习率 | 0.97 / 3e-4 | 0.97 / 3e-4 |
| 评估 | 21 次，每次 64 局，确定性策略 | 同左 |
| 每次采样后的 critic / actor 更新 | 1 / 1 | 8 / 2 |

**1 亿步是所有环境累计的 transition，不是每个环境 1 亿步，也不是物理子步。**
64 个环境对应约 1,562,500 次采样。它是统一的长训练预算，远大于过去的 1.5M/3M；
并行不会让相同的 timestep 变成更多样本。10 组总预算约为 10 亿 transition。
保留 step 0 和大约每 5M 步的评估，可查看 5M、10M、20M 等阶段的学习曲线。
Brax 按完整评估区间向上取整，在本配置下实际结束于 100,000,512 步；
FastSAC 结束于 100,000,000 步，这个微小差异应按日志中的实际步数比较。

[NVIDIA 规格](https://www.nvidia.com/content/nvidiaGDC/zz/en_ZZ/geforce/graphics-cards/rtx-2080-ti.html)
列出的 RTX 2080 Ti 显存为 11GB。原先的 256 环境在服务器 FastSAC 采样阶段发生 Warp 显存分配失败，
现将训练环境降为 **64**，评估也降为 **64**；新配置仍待服务器验证。接触/约束、完整重置、JIT 临时缓冲、
评估和网络反向传播也需要显存，尤其 Reorient 的多形状接触应留余量。

回放数组可以静态估算：若观测维数为 D、动作维数为 A，Brax 约为
`262144 × 4 × (2D + A + 3)` 字节，FastSAC 约为
`262144 × [4 × (2D + A + 1) + 2]` 字节。例如 D=200、A=40 时约 0.43 GiB，
**这只是 replay，不是训练峰值显存**。两个入口会打印各任务的实际 replay
数组估算。本次未启动 GPU 或仿真，也没有给出实测峰值或完成时间。

脚本关闭 JAX 预分配，让 JAX 与 Warp 按需申请显存；这不是显存上限，
也不保证不会 OOM。[JAX 显存说明](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)。
当前每环境接触/CCD 容量仍为 256，CCD 迭代仍为 150；不裁减碰撞容量或精度。
报错中的 `epa_pr` 单数组申请为 594,542,592 字节（567 MiB），它来自
`256环境 × 256容量 × (6 + 5 × 150) × 12字节`。
改为 64 环境后同一数组约为 141.75 MiB，其他按环境增长的物理缓冲也随之缩小；
仍不能把这一数组大小当作总峰值显存。Batch 同步从 1024 降为 256，保持
`batch × 每次采样的更新次数 / 环境数`：SAC 为 4、FastSAC 为 32。
FastSAC 的回放改为每环境 4096 条、预热 128 次采样，总容量及预热样本量不变。

FastSAC 在服务器降到 64 环境后仍于采样阶段发生显存不足，因此还限制执行图缓存：
`--warp_graph_cache_size=1`（默认值）。MJX 3.6 内置的 Warp FFI 原先默认按
输入/输出地址缓存最多 32 份 CUDA graph；缓存图会保留其所需的临时分配。
这里在创建环境及 JIT 前设置**实际被 MJX 使用的 vendored FFI**缓存上限，
不修改 site-packages，也不改 Brax SAC 的进程设置。这个上限是每个 FFI callable
的图数量，不是全进程显存上限，捕获新图时还可能短暂与旧图共存。

FastSAC 还在每次采样和 learner 更新后调用 `jax.block_until_ready`，限制异步
排队；不会把 replay 或物理状态复制回 CPU。以上措施保留碰撞容量、碰撞精度、
回放、网络和算法参数，但可能因重新捕获图及减少重叠执行而降低吞吐。
这是根据当前依赖源码定位的显存放大点，仍需服务器实测确认能否消除 OOM。
启动 FastSAC 时应看到 `MJX-Warp graph cache: 1 per FFI callable`。

如仍超显存，应先检查该物理 GPU 是否有其他进程以及实际峰值。
已经正常运行的旧配置不用中断，但其结果需标记为旧配置，不能与新配置当作
完全一致的参数实验。每个脚本每条 Python 命令都是新进程；重新启动队列时，
注意跳过已经完成的命令。

Brax SAC 的评估环境单独创建，按评估环境数分配 Warp 缓冲。
另加第一局指标筛选：已结束环境之后产生的 NaN/Inf 不再经由 `0 × NaN`
污染第一局统计。八个 manipulation 环境另有下文的 NaN/Inf 隔离与异常终止；
未被环境隔离处理的非有限指标仍会报错并保存 `nonfinite_metrics.json`。
这些处理不等于已经验证物理稳定性，应同时查看数值异常比例。

两者同一任务使用相同的 learner 奖励缩放：KeyTurn、Reorient（含 Die）为
0.01，PenTwirl 也为 0.01，Pose、Reach、Baoding 为 1.0；
记录的 episode reward 保持原始尺度。
FastSAC 的回报分布范围仍为 [-20,20]，需关注 `training/support_clip_fraction`。
网络结构、Q 表示、优化器、熵目标、tau 和延迟更新保留各算法现有配置，
因此这是**两套训练方法在相同样本预算下的比较，不是仅改变一项的消融实验**。
FastSAC 的更新计算量更大；两者均记录 `experiment/walltime`（秒，含初始化、
编译和评估），同时比较成功率随总步数和实际时间的变化。
单 seed 结果用于初步比较，不代表多 seed 的统计结论。

### 三个脚本与运行顺序

| GPU | 脚本 | 依次运行 |
| --- | --- | --- |
| 0 | [train_sac_comparison_2080ti_gpu0.sh](train_sac_comparison_2080ti_gpu0.sh) | KeyTurnRandom SAC → FastSAC；FingerPoseRandom SAC → FastSAC |
| 1 | [train_sac_comparison_2080ti_gpu1.sh](train_sac_comparison_2080ti_gpu1.sh) | Reorient100 SAC → FastSAC；HandReachRandom SAC |
| 2 | [train_sac_comparison_2080ti_gpu2.sh](train_sac_comparison_2080ti_gpu2.sh) | PenTwirlRandom SAC → FastSAC；HandReachRandom FastSAC |

分配为 4 / 3 / 3 次，其中 FastSAC 为 2 / 1 / 2 次；每张卡都混合两种算法。
这是按更新计算量做的初步均衡，没有各任务实测耗时，不能保证同时结束。
在老服务器把整个仓库和可用的依赖环境准备好，W&B 已登录后，分别在三个
终端或 tmux 窗口运行（下面是三条独立启动命令）：

```bash
bash myosuite/envs/myo/mjx/train_sac_comparison_2080ti_gpu0.sh
bash myosuite/envs/myo/mjx/train_sac_comparison_2080ti_gpu1.sh
bash myosuite/envs/myo/mjx/train_sac_comparison_2080ti_gpu2.sh
```

每个脚本按自身位置定位仓库的 `.venv`，不硬编码 `/home/lsy`，并显式设置
`CUDA_VISIBLE_DEVICES=0/1/2`。进程内只看到一张卡；Brax 不会抢占另外两张。
两个入口的 W&B project 均为 **myosuite**，run name 示例：

```text
MjxHandKeyTurnRandom-v0-sac-0424-1837
MjxHandKeyTurnRandom-v0-fastsac-0424-1837
```

时间为运行机器的本地时间。两者的本地保存目录附加更精细的时间后缀，
避免同一分钟重启时覆盖旧结果。此处只准备脚本，不自动启动训练。

### W&B：统一的奖励与成功率面板

两种算法均在独立的 **`metrics`** 分组上传训练过程的标量历史曲线，横轴统一为
`env_steps`。每 100 次并行采样（当前为 6,400 个环境步）统计本窗口内已完成的
episode；尚未结束的局跨日志窗口继续累计，结束后才计入。窗口内没有完成的局时
不伪造奖励或成功率数据点。预热期间也统计，评估和预热边界可能额外输出一次。
这套成功统计适用于所有受支持任务，包括暂不排期的任务，不只当前五个。

| 指标 | 两种算法共同的含义 |
| --- | --- |
| `metrics/reward` | 本日志窗口内已完成训练 episode 的原始累计奖励均值，不含 learner 奖励缩放 |
| `metrics/success_rate` | 本窗口内已完成训练 episode 中至少成功过一次的比例，范围 0～1 |
| `metrics/solved_step_fraction` | 每局成功时刻数 / 实际局长，再对各局求均值 |
| `metrics/numerical_failure_rate` | 数值异常终止的 episode 比例；环境提供该诊断时记录 |

保留原有 `eval/*`、`training/*` 诊断指标。确定性策略评估另上传到
`eval_metrics/reward`、`eval_metrics/success_rate`、`eval_metrics/solved_step_fraction`
及 `eval_metrics/numerical_failure_rate`，仍保持原评估频率。训练和评估数据不混线，
不重复填入旧评估值。配置记录 `metrics_source=completed_training_episodes`，
区别于旧版仅将评估结果映射到 `metrics/*` 的 run。Pen/Pose/Reach 的
`solved_frac` 不能直接当作“本局曾成功率”；Brax 已补齐与 FastSAC 相同的首次成功统计。
例如一局共 100 步、其中 5 步成功：该局成功时刻占比为 0.05，本局成功标记为 1。
`success_rate` 再对本窗口已完成的所有局取平均，两项均以 0～1 的比例上传。

上传方式与原 Pen/Pose/Reach 的训练入口一样，使用 `run.log(标量字典, step=环境步数)`
逐点追加 history；显式指定横轴，并关闭这两组指标的 summary 聚合，避免将最终值
当作训练曲线。新 run 在完成几个 episode 后应有多个训练数据点。
若已有 workspace 保存了条形图面板，需要在 W&B 将该面板改为 **Line plot**，
X 选 `env_steps`，Y 选对应 `metrics/*`；代码的指标定义不会重写已保存的面板配置。
旧 run 如果只上传过 step 0，无法凭空补出后续曲线；正在运行的进程也不会热加载代码。

### 其他任务：保留支持，暂不纳入排期

下列环境均可使用 `train_jax_sac.py` 或 `train_jax_fastsac.py`，但三个脚本不会运行它们：

| 类别 | 其他环境 ID |
| --- | --- |
| KeyTurn | `MjxHandKeyTurnFixed-v0` |
| SAR Reorient | `MjxHandReorient8-v0` |
| Die Reorient | `MjxChallengeDieReorientP1-v0`、`MjxChallengeDieReorientP2-v0` |
| Baoding | `MjxChallengeBaodingP1-v1`、`MjxChallengeBaodingP2-v1` |
| PenTwirl | `MjxHandPenTwirlFixed-v0` |
| Pose | `MjxFingerPoseFixed-v0`、`MjxElbowPoseFixed-v0`、`MjxElbowPoseRandom-v0` |
| Reach | `MjxHandReachFixed-v0`、`MjxHandReachRandomCustom-v0` |

例如手动运行肘部随机姿态（不在本轮队列）：

```bash
python train_jax_sac.py --env_name=MjxElbowPoseRandom-v0 --impl=warp --num_envs=64 --seed=42 --log_to_wandb --wandb_project=myosuite
python train_jax_fastsac.py --env_name=MjxElbowPoseRandom-v0 --impl=warp --num_envs=64 --seed=42 --log_to_wandb --wandb_project=myosuite
```

下面原有八个 manipulation 任务的命令继续保留作参考，**不是当前排期清单**。

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
python train_jax_sac.py --env_name=MjxHandKeyTurnFixed-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_sac.py --env_name=MjxHandKeyTurnRandom-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_sac.py --env_name=MjxHandReorient8-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_sac.py --env_name=MjxHandReorient100-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_sac.py --env_name=MjxChallengeDieReorientP1-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_sac.py --env_name=MjxChallengeDieReorientP2-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_sac.py --env_name=MjxChallengeBaodingP1-v1 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_sac.py --env_name=MjxChallengeBaodingP2-v1 --impl=warp --num_envs=64 --log_to_wandb
```

PPO 使用环境的 `{"state": observation}`，SAC 入口通过
`FlatStateObservationWrapper` 取出数组，以匹配本地 Brax SAC 接口。
PPO 仍为 10M 步、16 次评估；SAC 为 100M 步、21 次评估（含 step 0），
SAC 每次 64 个 episode（PPO 仍为 128）。SAC 默认为确定性策略评估，seed 42。
SAC 资源参数可以在命令行覆盖，`python train_jax_sac.py --help` 不初始化 GPU。

SAC 和 FastSAC 默认均为 `--log_interval=100`：每 100 次并行采样输出终端日志、
写入 `metrics.jsonl`，并在启用 W&B 时上传。当前 64 个环境、action repeat 为 1，
即每 **6,400 个环境步**一次；评估完成时也会记录，SAC 预填充结束时额外记录一次。
预填充期间显示 `replay warmup`、步数和 SPS；开始更新后显示 critic/actor loss、
alpha 和 SPS。`training/vector_steps`、`training/gradient_steps` 分别记录采样次数
和梯度更新次数。训练 loss 是本次分段内更新的均值。
`eval/episode_reward`、成功率等评估指标仍仅在原来的 21 次评估时更新
（约每 500 万环境步一次），不会每 6,400 步重新评估。

为了将训练日志与评估频率分开，SAC 入口使用本目录的 `brax_sac_train.py`：
基于 Brax 0.14.1 的 Apache-2.0 训练循环，保留原作者版权与许可说明。
仅将预填充和训练循环分段回传进度；分段之间保留环境、replay、优化器和随机数状态，
网络、损失和更新规则仍使用 Brax。分段长度是动态参数，避免为不同长度重复编译。
评估次数、checkpoint 时机和总训练步数保持原配置；不需要修改或重新安装 Brax。
首次编译期间仍可能暂时没有新日志。正在运行的 Python 进程不会自动加载此修改，
新日志频率从下一次启动训练时生效。

### 环境 NaN / Inf 处理

八个手部 manipulation 环境以及 PenTwirl、Pose、Reach（含 Custom Reach）
均在 reset/step 输出处使用 `nan_to_num`，
显式将 NaN、正负 Inf 替换为 0（不用默认的浮点最大值）。
若观测、奖励、指标或物理状态 `qpos/qvel/qacc/act` 出现非有限值，
该步的观测、奖励和原任务指标全部清零，设 `done=1`，由训练包装器完整重置。
正常步保持不变；该处理同时用于 SAC 和 FastSAC，避免异常输出进入观测统计与 replay。
Pen/Pose/Reach 的 reset 也先调用 `mjx.forward`，计算物体、指尖位置等派生量，
再构造初始观测。此前只有八个 manipulation 环境接入了异常隔离；旧版转笔训练
仍可能将异常仿真输出写入 replay。同步此修复后需启动新进程，旧 replay/模型不会自动修复。
直接调用环境、未使用训练包装器时，调用者必须在 `done` 后 reset。

W&B 的 `eval/episode_numerical_failure` 表示评估 episode 因数值异常终止的比例。
Brax SAC 另记录 `training/numerical_failure_per_step`（日志窗口内异常环境步比例）；
FastSAC 记录 `training/episode_numerical_failure`（窗口内已结束 episode 的异常比例）。
Brax 评估还记录 `nonfinite_observation/physics/reward` 的 episode 指标以区分异常来源。
异常步不记成功，但异常发生前已经记录的成功仍按“本局曾成功”定义保留。

这是异常隔离措施，尚未通过 GPU 运行确认 Reorient100/DieReorientP2 的根因已解决。
异常终止会改变 episode 长度和回报，比较算法时应同时查看异常比例；频繁出现时
仍需排查物理稳定性。它也不能修复已有 NaN 的网络参数、有限但过大的观测，
或梯度更新自身产生的 NaN；训练 loss 的非有限值检查仍保留。请从新 run 启动。

PPO/SAC 均写入 W&B **`myosuite` project**；不传 `--log_to_wandb` 则关闭。
SAC 名称为 `环境名-sac-MMDD-HHMM`；PPO 的命名维持原样。
SAC 本地保存到 `runs/sac/<run名>-<唯一时间后缀>/`，包含配置、指标 JSONL、
每轮评估前的 Brax policy checkpoint；这些 checkpoint 不包含 replay 或完整
优化器状态。`--save_policy` 另保存最终 `playground_params.pickle` 到该 run
目录，不再覆盖工作目录中的同名文件。

## FastSAC：Holosoma 算法的 JAX 适配

新增入口 **`train_jax_fastsac.py`**，使用本目录的环境和 MJX-Warp 物理，
训练部分为 JAX/Flax/Optax。它保留 Holosoma FastSAC 的核心更新逻辑，
不是调用 Brax 普通 SAC，也不依赖整个 Holosoma 安装。
代码来源、版本及移植差异见 [fast_sac/NOTICE.md](fast_sac/NOTICE.md)。
**本次仅完成源码实现和静态检查，未运行环境、JIT、数值测试或训练。**

### 依赖选择

继续使用 `/home/lsy/pycode/myosuite/.venv`，**当前无需安装额外依赖**。
没有执行 `pip install`、`uv sync` 或改变已安装版本。不要为这个入口在该
venv 中执行 Holosoma 的完整安装脚本：

| 依赖/组件 | 本地 Holosoma 的要求或安装行为 | 当前 MyoSuite `.venv` |
| --- | --- | --- |
| NumPy | `>=1.23.5,<2` | `2.2.6`，完整安装会要求降级 |
| W&B | 固定 `0.22.0` | `0.25.1`，完整安装会要求切换版本 |
| MuJoCo | setup 脚本安装 `>=3.0.0` 及未锁版本的 `mujoco-warp[cuda]` | MuJoCo / MJX 均为 `3.6.0`，仓库要求 `<3.7` |
| Warp | 包要求 `>=1.10`，新物理包可能继续升级 | `1.11.1`，也是本地 MJX 3.6.0 的 warp extra 指定版本 |
| 训练框架 | PyTorch、TensorDict 及其 CUDA 依赖 | JAX/JAXlib `0.6.2`、Flax `0.10.7`、Optax `0.2.8` |
| 其他组件 | Open3D、ONNX、机器人相关依赖等 | FastSAC 的 JAX 移植不需要 |

上述是 2026-09-20 读取本地包 metadata 和 Holosoma 源码的结果；并不表示
NumPy 1.x 与 JAX 一定不兼容，而是完整安装会改变现有已使用的依赖组合。
现有 MJX 3.6.0 的 Warp 支持随其依赖提供，无需为了本入口单独安装最新
`mujoco-warp`。Python 仍使用此 venv 的 3.10.19。

下面的检查入口只读包 metadata，不 import JAX/MuJoCo、不初始化 GPU：

```bash
cd /home/lsy/pycode/myosuite/myosuite/envs/myo/mjx
source /home/lsy/pycode/myosuite/.venv/bin/activate
python train_jax_fastsac.py --check_dependencies
```

### 八个任务的命令

后续 GPU 空闲时选择一条运行。默认使用一个 GPU；用 `CUDA_VISIBLE_DEVICES`
控制可见设备，`--device` 为其中的本地编号。若没有 GPU 默认报错，不会悄悄
改用 CPU。示例并行数和超参数尚未测量显存或调优。

```bash
python train_jax_fastsac.py --env_name=MjxHandKeyTurnFixed-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_fastsac.py --env_name=MjxHandKeyTurnRandom-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_fastsac.py --env_name=MjxHandReorient8-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_fastsac.py --env_name=MjxHandReorient100-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_fastsac.py --env_name=MjxChallengeDieReorientP1-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_fastsac.py --env_name=MjxChallengeDieReorientP2-v0 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_fastsac.py --env_name=MjxChallengeBaodingP1-v1 --impl=warp --num_envs=64 --log_to_wandb
python train_jax_fastsac.py --env_name=MjxChallengeBaodingP2-v1 --impl=warp --num_envs=64 --log_to_wandb
```

也支持本地已注册的 Pose、Reach、PenTwirl 环境；八个新任务可以使用表格中的
CPU 名称别名。转笔示例：

```bash
python train_jax_fastsac.py --env_name=MjxHandPenTwirlRandom-v0 --impl=warp --num_envs=64 --log_to_wandb
```

### 参数、回放和算法差异

- `--num_timesteps=100000000`：默认总计 100M 个**控制步 transition**，跨所有
  并行环境计数，向上取整到完整 vector step；物理子步由环境自己执行。
- `--buffer_size=4096`：**每个环境**存 4096 个控制步；64 个环境合计 262144
  条 transition。回放存原始观测及重置前 next_obs，启动时会打印预计占用。
- `--batch_size=256`：每次梯度更新的总样本数，不再乘 `num_envs`。
  `--num_updates=8`：每个 vector step 后更新 critic 8 次。
  `--policy_frequency=4`：每 4 次 critic 更新做一次 actor 更新。
- `--learning_starts=128`：先收集 128 个 vector step，即 8192 条 transition。
  `--num_steps=1` 是 n-step return 的 n，可以设为 3 等；它不是 frame_skip。
  buffer_size 和 learning_starts 都必须不小于 n。
- 保留离散回报分布表示（categorical distributional critic）、各 critic
  自己的 target distribution、actor 使用 Q 集合均值、LayerNorm/SiLU、
  tanh 高斯策略、自动温度调节、AdamW 和 Polyak 更新。策略动作保持
  `[-1,1]`，沿用环境内部的肌肉 sigmoid 映射，不再额外缩放一次。
- 默认 `gamma=0.97`、`tau=0.125`、`num_atoms=101`、`v_min=-20`、`v_max=20`、
  `alpha_init=0.001`、`target_entropy_ratio=0.0` 来自上游通用配置，
  **并非 MyoSuite 调优结果**。
- `--reward_scale` 默认按任务选择：KeyTurn、Reorient（含 Die）、PenTwirl
  为 `0.01`，其他任务（含 Baoding）为 `1.0`。缩放仅用于学习；日志里的
  episode reward 始终是环境原始奖励。要观察 `training/support_clip_fraction`：
  它是 target distribution 投影前越过支持区间的概率质量比例；持续较高时
  需调整 `--v_min`、`--v_max` 或 `--reward_scale`，不能照搬所有任务的范围。
- 回放在 termination/truncation 两种边界均停止累加；真正终止不 bootstrap，
  时间截断从本局最后观测 bootstrap。不会使用下一局 reset 观测来计算上一局
  的 Q target，也不会跨越 ring buffer 最新写入位置拼接 n-step 数据。
- FastSAC 使用独立的完整重置适配器，不经过 Brax 的 auto-reset wrapper。
  当一个 batch 中有环境结束时，生成候选 reset batch 并选择结束的环境，
  同步重置全部 data/obs/info；没有环境结束时跳过候选重置。
- 当前为**单设备、float32、MLP**实现，没有移植 PyTorch AMP、分布式训练、
  CNN、机器人对称增强或 ONNX 导出；不能据此承诺与上游相同的速度/学习曲线。

完整参数用 `python train_jax_fastsac.py --help` 查看；参数同时接受下划线和
连字符拼写。例：`--no-obs-normalization` 关闭观测归一化。

### 日志与 checkpoint

FastSAC 默认 seed 42，写入 W&B `myosuite` project，run 名为
`环境名-fastsac-MMDD-HHMM`；可通过
`--wandb_project` / `--wandb_entity` 修改。不传 `--log_to_wandb` 时无需联网。
本地始终保存 `config.json` 和 `metrics.jsonl`，默认目录为当前工作目录下
`runs/fastsac/<run名>-<唯一时间后缀>/`，可用 `--log_dir` 指定根目录。

- 默认 21 次评估（含 step 0 和训练结束），每次 64 个新 episode，使用确定性策略并冻结观测归一化。
  `eval/episode_solved_frac`、`eval/episode_solved_per_step`、
  `eval/episode_success` 的定义与下文一致；每个评估环境只统计第一局。
- `training/episode_*` 是**本次日志窗口内已完成 episode**的均值，
  `training/episodes_in_window` 给出样本数；不是固定最近几百局的滑动均值。
  没有 episode 完成时不输出对应均值。
- `training/sps` 是完成的控制 transition / 秒，包含采样和梯度更新，
  首个窗口包含 JIT 时间；评估时间单列 `eval/walltime`。
  `training/actor_updates` 是最近一组更新中的 actor 更新数，
  `training/gradient_steps` 是累计 critic 更新数。
- 默认每 78125 个 vector step（64 环境下 5M transition） 及结束时保存 `.msgpack` 和对应 `.json`。
  `--save_interval=0` 仅关闭中间保存，结束仍保存；不需要 `--save_policy`。
  参数、优化器、归一化状态均保留，但**不保存回放和物理环境状态**。

加载已有 checkpoint 继续优化（warm start；重新填充回放并启动新的统计）：

```bash
python train_jax_fastsac.py --env_name=MjxHandKeyTurnFixed-v0 --impl=warp --num_envs=64 \
  --load_checkpoint=/path/to/step_000010000384.msgpack --log_to_wandb
```

加载时需要保持任务、观测/动作维度、网络结构、Q 支持区间、奖励缩放等一致。
这不是精确断点续训，也不兼容 Holosoma `.pt` 或旧 Brax pickle。
推理可用 `fast_sac.checkpoint.load_policy(path)`，返回 `(policy, metadata)`；
`policy(state.obs["state"])` 输出传给原环境 `step` 的动作。通用可视化脚本
尚未适配此格式及逐局随机 model，不能直接拿新文件替换旧播放器的 pickle。

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
   SAR 的物体/目标方向向量按本局实际 marker 间距归一化，参考 MyoSuite3
   PR #406；旧 CPU 实现一直用 XML 初始长度，会让方向幅值随物体尺寸变化。
   因此 `obj_rot`、`target_rot` 及两者之差的观测数值与旧版不同，维度、顺序、
   方向余弦奖励及成功阈值不变。旧观测尺度下训练的策略需要重新评估。
6. **可视化**：训练使用的是从 `State.info` 重建的动态 model。现有通用
   `visu_mjx_env.py`/静态 model 播放器尚未适配逐帧恢复这些随机模型参数，
   后续播放时需同时恢复模型参数，不能仅凭 qpos 还原随机物体和目标。

## 官方 MJX / MyoSuite3 源码对照

本次对照固定到官方 `mjx` 分支的
[`f38091b`](https://github.com/MyoHub/myosuite/tree/f38091baf5f6bc4f63a83ceb0887f08a82463f9d)
及 [PR #406](https://github.com/MyoHub/myosuite/pull/406) 的
[`832cf59`](https://github.com/MyoHub/myosuite/tree/832cf59efdb1b69014588eb7a706f8f93a1c95fb)。
这是源码审查，没有执行上游或本地的环境测试、仿真、编译或训练。

- **采用**：PR 中 SAR 在每次采样尺寸后重新计算方向归一化长度的处理，
  见 [Reorient8/100 的 CPU 实现](https://github.com/MyoHub/myosuite/blob/832cf59efdb1b69014588eb7a706f8f93a1c95fb/myosuite/envs/myo/tasks/basic/arm/reorient_sar.py#L265)。
  本地直接由当前世界坐标 marker 间距求单位方向，不缓存共享的可变长度。
- **已有对应处理**：重置后 forward、速度乘控制周期、独立配置工厂。
  保留本地完整 episode 重置，以同步刷新随机物体参数、目标、观测及成功历史。
- **未搬入**：PR 的模块化任务架构及通用碰撞预处理。其
  [预处理](https://github.com/MyoHub/myosuite/blob/832cf59efdb1b69014588eb7a706f8f93a1c95fb/myosuite/envs/myo/backends/mjx/mjx_spec_preprocess.py)
  对 JAX 关闭 cylinder/ellipsoid 接触，不适合直接用于依赖这些物体接触的任务。
  PR 的 [MJX 注册表](https://github.com/MyoHub/myosuite/blob/832cf59efdb1b69014588eb7a706f8f93a1c95fb/myosuite/envs/myo/backends/mjx/__init__.py)
  没有这八个操作任务的现成 MJX 注册，
  [后端说明](https://github.com/MyoHub/myosuite/blob/832cf59efdb1b69014588eb7a706f8f93a1c95fb/myosuite/envs/myo/backends/mjx/README.md)
  也将 MJX 列为实验性支持，主要支持的 GPU 路径为 mjlab。

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
- `train_jax_fastsac.py`：FastSAC 参数入口和无 GPU 的依赖 metadata 检查。
- `fast_sac/`：Flax 网络、FastSAC 更新、n-step 回放、MJX 适配、训练和 checkpoint。
