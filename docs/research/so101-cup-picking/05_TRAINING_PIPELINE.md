# RL Training Pipeline

## Overview

Isaac Lab supports multiple RL frameworks. For the SO-101 cup-picking task, we recommend:

| Framework | Strengths | When to Use |
|-----------|-----------|-------------|
| **RSL-RL** | Fast PPO, good for manipulation | Phase 1 (state-based) |
| **SKRL** | Flexible, good documentation | Alternative to RSL-RL |
| **RL Games** | Mature, LSTM support | Complex policies |

## Recommended Algorithm: PPO

**Proximal Policy Optimization (PPO)** is the standard choice for manipulation tasks in Isaac Lab. All existing lift/reach tasks use PPO.

### Why PPO for Cup Picking?

- Stable training with continuous action spaces
- Works well with dense reward signals (distance-based rewards)
- Supports parallel environments (GPU-accelerated)
- Good sample efficiency with thousands of parallel envs

## RSL-RL Configuration (Recommended)

Based on the existing OpenArm lift PPO config at:
`source/isaaclab_tasks/.../lift/config/openarm/agents/rsl_rl_ppo_cfg.py`

```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class SO101CupPickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24          # rollout length
    max_iterations = 3000           # total training iterations
    save_interval = 100             # save checkpoint every N iterations
    experiment_name = "so101_cup_pick"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],    # MLP layers
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,          # exploration bonus
        num_learning_epochs=5,       # PPO epochs per rollout
        num_mini_batches=4,          # mini-batch size
        learning_rate=1.0e-4,
        schedule="adaptive",         # adaptive LR based on KL divergence
        gamma=0.98,                  # discount factor
        lam=0.95,                    # GAE lambda
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### Key Hyperparameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `num_steps_per_env` | 24 | Short rollouts -- manipulation episodes are brief |
| `max_iterations` | 3000 | Enough for convergence with 4096 envs |
| `actor_hidden_dims` | [256, 128, 64] | 3-layer MLP -- sufficient for state-based policy |
| `entropy_coef` | 0.006 | Mild exploration -- too high causes random behavior |
| `learning_rate` | 1e-4 | Conservative LR with adaptive schedule |
| `gamma` | 0.98 | High discount -- future rewards matter for multi-stage task |
| `lam` | 0.95 | GAE lambda -- balance bias/variance in advantage estimation |

## Training Commands

### Phase 1: State-Based Training

```bash
# Standard training with visualization
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4096

# Headless (faster)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4096 \
    --headless

# With video recording
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4096 \
    --headless \
    --enable_cameras \
    --video \
    --video_interval 500

# Resume from checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4096 \
    --headless \
    --resume True \
    --load_run <run_directory_name>
```

### Evaluate / Play Trained Policy

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Pick-Cup-SO101-Play-v0 \
    --num_envs 50 \
    --load_run <run_directory_name>
```

### Multi-GPU Training

```bash
# Distributed training across multiple GPUs
./isaaclab.sh -p -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --distributed
```

## SKRL Alternative

If you prefer SKRL (which has excellent documentation and supports more algorithms):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4096 \
    --headless
```

SKRL configuration example:

```python
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4
cfg["discount_factor"] = 0.98
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-4
cfg["grad_norm_clip"] = 1.0
cfg["entropy_loss_scale"] = 0.006
```

## Training Logs and Monitoring

### Log Location

```
logs/rsl_rl/so101_cup_pick/<timestamp>/
  params/
    env.yaml          # Environment config snapshot
    agent.yaml        # Agent config snapshot
  model_*.pt          # Checkpoints
  exported/
    policy.onnx       # ONNX export (generated by play script)
```

### TensorBoard Monitoring

```bash
tensorboard --logdir logs/rsl_rl/so101_cup_pick/
```

Key metrics to watch:
- `Loss/value_function` - should decrease
- `Loss/surrogate` - should stabilize
- `Policy/mean_noise_std` - should decrease as policy becomes confident
- `Train/mean_reward` - should increase
- `Train/mean_episode_length` - should increase (agent keeps cup lifted longer)

## Expected Training Timeline

| Phase | Iterations | Expected Behavior |
|-------|-----------|-------------------|
| 0-500 | Random exploration | Arm flails randomly |
| 500-1000 | Reaching | Arm moves toward the cup |
| 1000-1500 | Contact | Arm touches the cup, sometimes grasps |
| 1500-2000 | Grasping | Reliable grasp, begins lifting |
| 2000-3000 | Lifting | Consistent lift to target height |

With 4096 environments on a modern GPU (RTX 4090), each iteration processes ~100K environment steps. Total training wall time is typically 1-3 hours for 3000 iterations.

## Troubleshooting

### Issue: Arm Doesn't Move Toward Cup

- Check that observations include cup position
- Verify the reaching reward weight is sufficient
- Increase `init_noise_std` for more initial exploration

### Issue: Grasps But Drops Immediately

- Increase gripper stiffness in `ArticulationCfg`
- Add a grip-force reward or contact-based reward
- Check that `BinaryJointPositionActionCfg` close position fully closes on the cup

### Issue: Training Diverges

- Reduce learning rate (try 3e-5)
- Reduce `clip_param` (try 0.1)
- Check reward scale -- very large rewards cause instability

### Issue: Very Slow Training

- Use `--headless` mode
- Increase `num_envs` (if VRAM allows)
- Check that GPU utilization is high (`nvidia-smi`)
