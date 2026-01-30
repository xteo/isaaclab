# Isaac Lab Development Context

> This file provides essential context for AI assistants (Claude Code) working with the Isaac Lab repository.

## Repository Purpose

Isaac Lab is NVIDIA's GPU-accelerated robot learning framework built on Isaac Sim. It enables:
- **Reinforcement Learning**: Train policies for locomotion, manipulation, navigation
- **Imitation Learning**: Learn from demonstrations via behavior cloning, GAIL
- **Sim-to-Real Transfer**: Deploy trained policies to real robots

## Quick Commands

```bash
# Training (RSL-RL - recommended for locomotion)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK_NAME> --num_envs 4096

# Evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task <TASK_NAME> --load_run <TIMESTAMP>

# List all environments
./isaaclab.sh -p scripts/environments/list_envs.py

# Test environment with random actions
./isaaclab.sh -p scripts/environments/random_agent.py --task <TASK_NAME> --num_envs 16
```

## Project Structure

```
isaaclab/
├── source/
│   ├── isaaclab/               # Core framework
│   │   ├── envs/               # Environment base classes
│   │   ├── managers/           # MDP managers (obs, action, reward, term)
│   │   ├── assets/             # Articulation, rigid body assets
│   │   ├── sensors/            # Camera, contact sensors
│   │   └── actuators/          # Motor models
│   ├── isaaclab_tasks/         # Task/environment definitions
│   │   ├── manager_based/      # Manager-based envs (RECOMMENDED)
│   │   └── direct/             # Direct RL envs
│   ├── isaaclab_assets/        # Robot configurations
│   │   └── robots/             # ANYmal, Spot, H1, Franka, etc.
│   └── isaaclab_rl/            # RL framework wrappers
├── scripts/
│   ├── reinforcement_learning/ # Training scripts per framework
│   └── environments/           # Environment demos
└── docs/                       # Documentation
```

## Core Concepts

### Two Environment Workflows

1. **Manager-Based** (Recommended for most cases):
   - Modular: Separate managers for observations, actions, rewards, terminations
   - Location: `source/isaaclab_tasks/isaaclab_tasks/manager_based/`
   - Base class: `ManagerBasedRLEnv`

2. **Direct** (For performance-critical applications):
   - Monolithic: All logic in single class
   - Location: `source/isaaclab_tasks/isaaclab_tasks/direct/`
   - Base class: `DirectRLEnv`

### Configuration Pattern

All configurations use `@configclass` decorator:

```python
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    # Scene setup
    scene: MySceneCfg = MySceneCfg(num_envs=4096)

    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
```

### Reward Function Pattern

Custom rewards go in `mdp/rewards.py`:

```python
import torch
from isaaclab.envs import ManagerBasedRLEnv

def velocity_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for tracking velocity command."""
    command = env.command_manager.get_command("base_velocity")
    actual = env.scene["robot"].data.root_lin_vel_w[:, :2]
    error = torch.sum(torch.square(command - actual), dim=1)
    return torch.exp(-error / 0.25)
```

Reference in config:
```python
rewards: RewardsCfg = RewardsCfg(
    velocity_tracking=RewardTermCfg(
        func=mdp.velocity_tracking_reward,
        weight=1.5,
    ),
)
```

## Creating Custom Environments

### Step 1: Choose a Template
- Simple task: `manager_based/classic/cartpole/`
- Locomotion: `manager_based/locomotion/velocity/`
- Manipulation: `manager_based/manipulation/reach/`

### Step 2: Create Configuration
```python
@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    # Define scene with robot and objects
    scene: MySceneCfg = MySceneCfg()

    # Define observations (what policy sees)
    observations: ObservationsCfg = ObservationsCfg()

    # Define actions (what policy outputs)
    actions: ActionsCfg = ActionsCfg()

    # Define rewards (learning signal)
    rewards: RewardsCfg = RewardsCfg()

    # Define terminations (episode end conditions)
    terminations: TerminationsCfg = TerminationsCfg()
```

### Step 3: Register Environment
In `__init__.py`:
```python
import gymnasium as gym

gym.register(
    id="My-Custom-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "my_package:MyTaskEnvCfg",
    },
)
```

### Step 4: Train
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task My-Custom-Task-v0 --num_envs 4096
```

## Common Issues & Solutions

### Robot "Explodes" at Start
**Cause**: Collision mesh overlap or unstable initial pose
**Solution**:
```python
# Add joint limit penalty
joint_limits=RewardTermCfg(func=mdp.joint_limits_exceeded, weight=-1.0)

# Check collision geometry in USD file
# Reduce physics timestep if needed
```

### CUDA Out of Memory
**Cause**: Too many parallel environments
**Solution**:
```bash
# Reduce num_envs
--num_envs 2048  # Instead of 4096

# Or increase GPU buffers in config
sim: SimulationCfg = SimulationCfg(
    physx=PhysxCfg(
        gpu_heap_capacity=2**26,
    )
)
```

### NaN in Observations/Rewards
**Cause**: Physics instability or sensor misconfiguration
**Solution**:
```python
# Add observation clipping
observations: ObservationsCfg = ObservationsCfg(
    policy=ObservationGroupCfg(clip=(-100.0, 100.0))
)

# Reduce physics timestep
sim: SimulationCfg = SimulationCfg(dt=0.005)  # Default is 0.01
```

### Training Not Converging
**Cause**: Sparse rewards or wrong hyperparameters
**Solution**:
1. Add shaping rewards (alive bonus, progress reward)
2. Reduce learning rate to 1e-4
3. Increase batch size
4. Use curriculum learning

## MuJoCo Users Quick Reference

| MuJoCo | Isaac Lab |
|--------|-----------|
| `env.sim.data.qpos` | `scene["robot"].data.joint_pos` |
| `env.sim.data.qvel` | `scene["robot"].data.joint_vel` |
| `mj_step()` | `env.step()` |
| MJCF XML | USD + ArticulationCfg |
| Single env | Batched: `[num_envs, ...]` |
| CPU/GPU optional | GPU-native (PyTorch) |

## Available Robots

| Robot | Config Location | Task Examples |
|-------|-----------------|---------------|
| ANYmal C/D | `isaaclab_assets.robots.anymal` | Locomotion |
| Unitree Go2 | `isaaclab_assets.robots.unitree` | Locomotion |
| Unitree H1 | `isaaclab_assets.robots.unitree` | Humanoid |
| Franka Panda | `isaaclab_assets.robots.franka` | Manipulation |
| UR10 | `isaaclab_assets.robots.universal_robots` | Manipulation |
| Spot | `isaaclab_assets.robots.spot` | Locomotion |

## RL Frameworks Supported

| Framework | Script Location | Best For |
|-----------|-----------------|----------|
| RSL-RL | `scripts/reinforcement_learning/rsl_rl/` | Locomotion |
| SKRL | `scripts/reinforcement_learning/skrl/` | General purpose |
| RL Games | `scripts/reinforcement_learning/rl_games/` | High performance |
| Stable-Baselines3 | `scripts/reinforcement_learning/sb3/` | Simplicity |

## Key Documentation

- Tutorials: `docs/source/_static/tutorials/`
- API Reference: `docs/source/api/`
- Task Workflows: `docs/source/overview/core-concepts/task_workflows.html`
- Troubleshooting: `docs/source/refs/troubleshooting.html`

## Agent Guidance Documents

Detailed guidance for AI assistants available in:
- `docs/agent_guidance/ISAAC_LAB_STRUCTURE.md` - Full repository map
- `docs/agent_guidance/AGENT_STRATEGY.md` - Agent design patterns
- `docs/agent_guidance/WEB_RESEARCH_FINDINGS.md` - MuJoCo/skills research

---

*Isaac Lab Version: 0.42.25 | Last Updated: January 2025*
