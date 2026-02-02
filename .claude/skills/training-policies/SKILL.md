---
name: training-policies
description: Guide for training RL policies in Isaac Lab. Use when user wants to train, fine-tune, evaluate, or deploy policies using RSL-RL, SKRL, RL Games, Stable Baselines 3, or Ray RLlib.
---

# Training Policies in Isaac Lab

## Supported RL Frameworks

| Framework | Best For | Config Type |
|-----------|----------|-------------|
| RSL-RL | Locomotion, fast training | Python |
| SKRL | General purpose, flexibility | YAML |
| RL Games | High performance | YAML |
| Stable Baselines 3 | Simplicity, baselines | Python |
| Ray RLlib | Distributed, scaling | Python |

## Quick Start: RSL-RL Training

```bash
# Basic training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless

# With custom config overrides
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --max_iterations 2000 \
    --experiment_name my_experiment

# Resume training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --resume \
    --load_run <RUN_DIR>
```

## RSL-RL Configuration

Location: `<task>/config/<robot>/agents/rsl_rl_ppo_cfg.py`

```python
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class MyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training settings
    num_steps_per_env = 24        # Steps before update
    max_iterations = 1500         # Total training iterations
    save_interval = 50            # Checkpoint frequency
    experiment_name = "my_task"
    run_name = ""                 # Auto-generated if empty

    # Logging
    logger = "tensorboard"        # or "wandb"

    # Parallelization
    empirical_normalization = False

    # Policy network
    policy = RslRlPpoActorCriticCfg(
        init_noise_std = 1.0,
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        activation = "elu",       # elu, relu, tanh
    )

    # PPO algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.01,
        num_learning_epochs = 5,
        num_mini_batches = 4,
        learning_rate = 1e-3,
        schedule = "adaptive",    # adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.0,
    )
```

## SKRL Training

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless
```

YAML config location: `<task>/config/<robot>/agents/skrl_ppo_cfg.yaml`

```yaml
seed: 42

models:
  separate: False
  policy:
    class: GaussianMixin
    clip_actions: False
    clip_log_std: True
    min_log_std: -20.0
    max_log_std: 2.0
    initial_log_std: 0.0
    network:
      - name: net
        input: OBSERVATIONS
        layers: [256, 128, 64]
        activations: elu

agent:
  class: PPO
  rollouts: 24
  learning_epochs: 5
  mini_batches: 4
  discount: 0.99
  lambda: 0.95
  learning_rate: 1.0e-3
  learning_rate_scheduler: KLAdaptiveLR
  state_preprocessor: RunningStandardScaler

trainer:
  class: SequentialTrainer
  timesteps: 36000
  environment_info: log
```

## Stable Baselines 3 Training

```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-Velocity-Flat-Anymal-C-v0 \
    --num_envs 64 \
    --headless
```

Note: SB3 is CPU-based and slower. Use for baselines or simpler tasks.

## Evaluating Trained Policies

```bash
# Play with trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --checkpoint logs/<experiment>/<run>/model_<iter>.pt

# Record video
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --checkpoint logs/<experiment>/<run>/model_<iter>.pt \
    --video \
    --video_length 500
```

## Hyperparameter Tuning Guide

### Learning Rate

| Symptom | Adjustment |
|---------|------------|
| Reward plateaus early | Increase LR (2x-5x) |
| Reward oscillates | Decrease LR (0.5x-0.2x) |
| NaN in training | Decrease LR, check rewards |

### Network Architecture

| Task Type | Recommended Dims |
|-----------|-----------------|
| Simple (cartpole) | [64, 64] |
| Locomotion | [256, 128, 64] or [512, 256, 128] |
| Manipulation | [512, 256, 128] |
| Complex | [1024, 512, 256] |

### PPO-Specific

| Parameter | Effect |
|-----------|--------|
| `clip_param` | Higher = more aggressive updates |
| `entropy_coef` | Higher = more exploration |
| `num_learning_epochs` | Higher = more gradient steps |
| `num_mini_batches` | Higher = smaller batches, more variance |
| `desired_kl` | Target KL for adaptive LR |

### Reward Scaling

- Rewards should be roughly in [-10, 10] range
- Normalize rewards if magnitudes vary widely
- Use `reward_scaler` in RSL-RL config

## Monitoring Training

### TensorBoard

```bash
# Launch tensorboard
tensorboard --logdir logs/

# Key metrics to watch:
# - episode_reward_mean
# - policy_loss
# - value_loss
# - entropy
# - learning_rate (if adaptive)
# - episode_length_mean
```

### Common Training Issues

| Issue | Possible Causes | Solutions |
|-------|-----------------|-----------|
| Reward not improving | LR too low, reward bug | Check reward values, increase LR |
| Reward oscillates | LR too high | Decrease LR |
| Policy collapses | Entropy too low | Increase `entropy_coef` |
| Training unstable | Reward scale, physics | Normalize rewards, check physics |
| OOM error | Too many envs | Reduce `num_envs` |

## Multi-GPU Training

```bash
# With RSL-RL (uses all available GPUs)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 16384 \
    --device cuda:0
```

## Checkpointing & Resume

Checkpoints saved to: `logs/<experiment>/<run>/`

```
logs/
└── my_experiment/
    └── 2024-01-15_12-30-45/
        ├── model_0.pt        # Initial model
        ├── model_500.pt      # Checkpoint at iter 500
        ├── model_1000.pt
        ├── config.yaml       # Training config
        └── events.out.*      # TensorBoard logs
```

Resume from checkpoint:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --resume \
    --load_run logs/my_experiment/2024-01-15_12-30-45
```

## Exporting Policies

```bash
# Export to ONNX for deployment
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/export_policy.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --checkpoint logs/.../model.pt \
    --output policy.onnx
```

## Checklist for Successful Training

- [ ] Verify environment runs with random agent
- [ ] Check reward values are reasonable (not all zeros, not huge)
- [ ] Start with fewer envs (512-1024) to debug
- [ ] Watch TensorBoard for first 100 iterations
- [ ] Scale up envs once working (4096+)
- [ ] Save checkpoints frequently for long runs
- [ ] Document hyperparameters that work
