# Holosoma FastSAC attribution and adaptation

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

This directory contains a JAX/Flax adaptation of algorithm components from
[amazon-far/holosoma](https://github.com/amazon-far/holosoma), under the
Apache License, Version 2.0 (the full license is included in the MyoSuite
repository's root `LICENSE`). The upstream copyright notice above is retained
from Holosoma's `NOTICE`.

Reference revision: `bccd4d7451640a2800ddc77e469d911a84f91994`, inspected in the
local `/home/lsy/pycode/holosoma` checkout on 2026-09-20.

Source components:

- `src/holosoma/holosoma/agents/fast_sac/fast_sac.py`: actor MLP, bounded
  log-standard-deviation, tanh policy, categorical distributional Q ensemble.
- `src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py`: per-critic target
  distributions, ensemble-mean actor objective, temperature update, AdamW,
  delayed actor and Polyak target updates.
- `src/holosoma/holosoma/agents/fast_sac/fast_sac_utils.py`: replay/normalization
  design, rewritten as functional device-resident JAX state.
- `src/holosoma/holosoma/config_types/algo.py`: FastSAC algorithm defaults.

## Changes in this adaptation

All code here is adapted or newly written for MyoSuite; it is not an unmodified
copy of the Holosoma training stack, and does not import Holosoma or PyTorch.

- Flax/Optax replace PyTorch/TensorDict. One JAX device, float32 MLP training;
  no PyTorch AMP, CUDA graph runner, distributed training, CNN encoder,
  humanoid symmetry augmentation, motion retargeting or ONNX export.
- Preserve the shrinking hidden widths (H, H/2, H/4), LayerNorm (epsilon 1e-5),
  SiLU, zero-initialized actor output heads, log-std bounds, individual
  distributional critic targets, ensemble-mean actor value, and exponential
  alpha loss. Hidden-layer initialization uses Flax defaults, not PyTorch's.
- Default gamma 0.97, tau 0.125, 101 atoms on [-20, 20], initial alpha 0.001,
  entropy ratio 0, 8 critic updates per collection step and 1 actor update per
  4 critic updates follow Holosoma's generic configuration. Actor/critic AdamW
  uses betas (0.9, 0.95), weight decay 0.001; alpha AdamW retains upstream's
  implicit default weight decay 0.01. These are starting points, not tuned
  MyoSuite hyperparameters or a claim of identical learning curves.
- Actor scheduling uses a continuous critic-update counter; it also works
  for policy_frequency=1 and numbers of updates not divisible by the frequency.
- Replay batch size is an exact TOTAL sample count, sampled uniformly across
  valid time/environment pairs. It is not rounded to a multiple of num_envs.
  Default capacity is 4096 steps per world (upstream generic default: 1024).
  The 2080 Ti comparison preset uses 64 worlds, 262144 total replay entries,
  batch size 256, and 128 warmup vector steps (8192 transitions). These resource
  defaults have not been validated on an RTX 2080 Ti.
- n-step returns stop at either termination or truncation. Both reward masking
  and the bootstrap discount use the actual number of steps; timeouts bootstrap
  from the pre-reset final observation. Sampling never crosses the ring's
  insertion seam. This deliberately corrects boundary handling rather than
  reproducing upstream replay behavior around truncation.
- Normalization uses the parallel Welford formula and updates once per newly
  collected observation rather than repeatedly on sampled replay batches.
  Replay stores raw observations; evaluation freezes normalization statistics.
- MyoSuite's normalized [-1, 1] actions are sent directly to env.step, which
  retains the existing muscle sigmoid mapping. No humanoid PD controller or
  joint-based action rescaling is introduced.
- Learner reward scaling defaults to 0.01 for KeyTurn/Reorient/PenTwirl and
  1.0 otherwise, to bring the very different task rewards closer to the
  categorical support. Reported episode returns always use original rewards.
  Monitor support_clip_fraction and tune support/scaling for each experiment.
- Independent full resets preserve per-episode task parameters; explicit
  pre-reset transitions make this usable without Brax's auto-reset wrapper.
- Checkpoints contain Flax learner/optimizer and normalization state, plus a
  JSON description. Loading warm-starts training with fresh replay/episodes;
  it is not an exact continuation of a previous run and cannot load .pt files.

Only source/static checks and package metadata inspection were performed during
implementation. No environment execution, JIT compilation, training or numerical
equivalence tests have been run.

## MJX-Warp graph memory policy

The FastSAC entry point sets MJX 3.6's vendored Warp FFI address-keyed graph
cache limit to 1 per callable before tracing (upstream default: 32). Collection
and update outputs are synchronized each iteration to bound work in flight.
This may trade throughput for lower memory pressure; it changes neither the
physics configuration nor SAC losses. No GPU validation has been performed
for this memory change.
