# Isaac Lab Repository Structure & AI Agent Navigation Guide

> A comprehensive guide for AI agents to navigate and work with the Isaac Lab codebase effectively.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Repository Overview](#repository-overview)
3. [Core Architecture](#core-architecture)
4. [Key Directories Reference](#key-directories-reference)
5. [Environment Development Patterns](#environment-development-patterns)
6. [Configuration System](#configuration-system)
7. [Training Pipeline](#training-pipeline)
8. [Navigation Cheatsheet](#navigation-cheatsheet)
9. [Instrumentation Recommendations](#instrumentation-recommendations)

---

## Executive Summary

Isaac Lab is NVIDIA's GPU-accelerated framework for robot learning built on Isaac Sim. The repository follows a **modular extension-based architecture** with six main packages:

| Package | Purpose | Key Location |
|---------|---------|--------------|
| `isaaclab` | Core framework | `/source/isaaclab/` |
| `isaaclab_tasks` | Environment suite | `/source/isaaclab_tasks/` |
| `isaaclab_assets` | Robot/asset configs | `/source/isaaclab_assets/` |
| `isaaclab_rl` | RL framework integration | `/source/isaaclab_rl/` |
| `isaaclab_mimic` | Imitation learning | `/source/isaaclab_mimic/` |
| `isaaclab_contrib` | Community contributions | `/source/isaaclab_contrib/` |

---

## Repository Overview

```
/home/user/isaaclab/
├── source/                    # Source code (6 extensions)
│   ├── isaaclab/              # Core framework v0.54.2
│   ├── isaaclab_tasks/        # Task implementations v0.11.12
│   ├── isaaclab_assets/       # Robot configurations
│   ├── isaaclab_rl/           # RL training integration
│   ├── isaaclab_mimic/        # Imitation learning
│   └── isaaclab_contrib/      # Community contributions
├── scripts/                   # Example scripts and tools
│   ├── tutorials/             # Step-by-step tutorials
│   ├── demos/                 # Demonstration scripts
│   ├── reinforcement_learning/# RL training scripts
│   ├── imitation_learning/    # IL training scripts
│   └── tools/                 # Utility scripts
├── docs/                      # Documentation source
├── tools/                     # Development templates
├── docker/                    # Docker configurations
├── apps/                      # Application configs
├── isaaclab.sh               # Main launcher (Linux)
├── isaaclab.bat              # Main launcher (Windows)
└── pyproject.toml            # Project configuration
```

---

## Core Architecture

### 1. Manager-Based Environment System

Isaac Lab uses a **manager-based architecture** where MDP components are handled by specialized managers:

```
ManagerBasedRLEnv
├── ActionManager          # Processes agent actions → robot commands
├── ObservationManager     # Generates observations from scene state
├── RewardManager          # Computes reward signals
├── TerminationManager     # Determines episode termination
├── EventManager           # Handles resets, randomization
├── CommandManager         # Generates goals/commands (optional)
├── CurriculumManager      # Schedules task difficulty (optional)
└── RecorderManager        # Records data for analysis
```

**Key File:** `/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`

### 2. Two Environment Paradigms

| Paradigm | Description | Best For |
|----------|-------------|----------|
| **Manager-Based** | Modular, multi-file, configurable | Complex tasks, research |
| **Direct** | Single-file, explicit control | Simple tasks, beginners |

**Manager-Based Location:** `/source/isaaclab_tasks/isaaclab_tasks/manager_based/`
**Direct Location:** `/source/isaaclab_tasks/isaaclab_tasks/direct/`

### 3. Configuration Class System

All configuration uses the `@configclass` decorator (dataclass-based):

```python
from isaaclab.utils import configclass

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg()
    actions: MyActionsCfg = MyActionsCfg()
    observations: MyObservationsCfg = MyObservationsCfg()
    rewards: MyRewardsCfg = MyRewardsCfg()
    terminations: MyTerminationsCfg = MyTerminationsCfg()
```

**Key File:** `/source/isaaclab/isaaclab/utils/configclass.py`

---

## Key Directories Reference

### For Creating New Environments

| Task | Primary Location |
|------|------------------|
| Environment base classes | `/source/isaaclab/isaaclab/envs/` |
| Built-in MDP functions | `/source/isaaclab/isaaclab/envs/mdp/` |
| Locomotion tasks | `/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/` |
| Manipulation tasks | `/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/` |
| Classic control tasks | `/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/` |

### For Working with Robots

| Task | Primary Location |
|------|------------------|
| Robot configurations | `/source/isaaclab_assets/isaaclab_assets/robots/` |
| Asset base classes | `/source/isaaclab/isaaclab/assets/` |
| Actuator models | `/source/isaaclab/isaaclab/actuators/` |
| Sensor configurations | `/source/isaaclab_assets/isaaclab_assets/sensors/` |
| USD asset files | `/source/isaaclab_assets/data/Robots/` |

### For Training Policies

| Task | Primary Location |
|------|------------------|
| RSL-RL training | `/scripts/reinforcement_learning/rsl_rl/` |
| SKRL training | `/scripts/reinforcement_learning/skrl/` |
| RL Games training | `/scripts/reinforcement_learning/rl_games/` |
| Stable Baselines 3 | `/scripts/reinforcement_learning/sb3/` |
| Agent configs | `/source/isaaclab_tasks/.../agents/` |

### For Learning Tutorials

| Task | Primary Location |
|------|------------------|
| Simulation basics | `/scripts/tutorials/00_sim/` |
| Asset management | `/scripts/tutorials/01_assets/` |
| Scene creation | `/scripts/tutorials/02_scene/` |
| Environment creation | `/scripts/tutorials/03_envs/` |
| Sensors | `/scripts/tutorials/04_sensors/` |
| Controllers | `/scripts/tutorials/05_controllers/` |

---

## Environment Development Patterns

### Pattern 1: Manager-Based Environment Structure

```
my_task/
├── __init__.py                 # Gym registration
├── my_task_env_cfg.py          # Environment configuration
├── config/
│   └── my_robot/
│       ├── __init__.py         # Robot-specific registration
│       ├── flat_env_cfg.py     # Flat terrain variant
│       ├── rough_env_cfg.py    # Rough terrain variant
│       └── agents/             # Training configs
│           ├── rsl_rl_ppo_cfg.py
│           ├── skrl_ppo_cfg.yaml
│           └── rl_games_ppo_cfg.yaml
└── mdp/
    ├── __init__.py
    ├── actions.py              # Custom action terms
    ├── observations.py         # Custom observation terms
    ├── rewards.py              # Custom reward functions
    ├── commands.py             # Custom command generators
    └── events.py               # Custom event handlers
```

### Pattern 2: Gym Registration

```python
# In __init__.py
import gymnasium as gym

gym.register(
    id="Isaac-MyTask-MyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env_cfg:MyTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyTaskPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

### Pattern 3: MDP Term Definition

```python
# In rewards.py
def my_custom_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """Computes reward based on custom criteria."""
    asset = env.scene[asset_cfg.name]
    # Custom reward computation
    return reward_tensor
```

---

## Configuration System

### Environment Config Hierarchy

```
ManagerBasedEnvCfg (base)
├── viewer: ViewerCfg
├── sim: SimulationCfg
│   ├── dt: float                    # Simulation timestep
│   ├── device: str                  # cuda:0, cpu
│   └── physics_material: PhysicsMaterialCfg
├── scene: InteractiveSceneCfg
│   ├── num_envs: int               # Parallel environments
│   ├── env_spacing: float          # Environment spacing
│   └── [entities...]               # Robots, objects, terrain
├── observations: ObservationsCfg
├── actions: ActionsCfg
├── events: EventsCfg
└── decimation: int                 # Action repeat

ManagerBasedRLEnvCfg (extends above)
├── rewards: RewardsCfg
├── terminations: TerminationsCfg
├── commands: CommandsCfg (optional)
├── curriculum: CurriculumCfg (optional)
├── episode_length_s: float
└── is_finite_horizon: bool
```

### Asset Configuration

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="path/to/robot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(...),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(...),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=80.0,
            damping=2.0,
        ),
    },
)
```

---

## Training Pipeline

### Running Training

```bash
# RSL-RL PPO training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless

# Play trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --checkpoint logs/.../model.pt
```

### Training Config Structure (RSL-RL)

```python
@configclass
class MyTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
    )
```

---

## Navigation Cheatsheet

### Quick Reference: "Where do I find...?"

| I want to... | Look here |
|--------------|-----------|
| Create a new environment | Copy from `/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/` |
| Add a new robot | Check `/source/isaaclab_assets/isaaclab_assets/robots/` for examples |
| Modify rewards | Edit `mdp/rewards.py` in your task folder |
| Add observations | Edit `mdp/observations.py` or use built-ins from `/source/isaaclab/isaaclab/envs/mdp/` |
| Configure actuators | See `/source/isaaclab/isaaclab/actuators/` |
| Add sensors | See `/source/isaaclab/isaaclab/sensors/` |
| Change training hyperparameters | Edit `agents/*_cfg.py` in your task config folder |
| Run imitation learning | See `/scripts/imitation_learning/` |
| Convert URDF to USD | Use `/scripts/tools/convert_urdf.py` |
| Debug physics | Run demos in `/scripts/demos/` with GUI |

### Common File Patterns

| Pattern | Meaning |
|---------|---------|
| `*_cfg.py` | Configuration file |
| `*_env_cfg.py` | Environment configuration |
| `mdp/*.py` | MDP component implementations |
| `agents/*.py` | Training agent configurations |
| `__init__.py` | Gym registration (in task folders) |

---

## Instrumentation Recommendations

### 1. Root-Level CLAUDE.md

Create `/home/user/isaaclab/CLAUDE.md` with:
- Quick command reference
- Architecture overview pointer
- Common workflow patterns
- File location hints

### 2. Per-Module AGENTS.md Files

Add in key directories:
- `/source/isaaclab_tasks/AGENTS.md` - Task development guide
- `/source/isaaclab_assets/AGENTS.md` - Asset management guide
- `/scripts/AGENTS.md` - Script usage guide

### 3. Claude Code Skills

Create skills for:
- `creating-environments` - Step-by-step environment creation
- `adding-robots` - Robot integration workflow
- `training-policies` - RL training guide
- `debugging-simulation` - Common debugging patterns
- `mujoco-translation` - MuJoCo to Isaac Lab translation

### 4. Code Comments Enhancement

Add inline navigation comments in key files:
```python
# NAVIGATION: For custom rewards, see mdp/rewards.py
# NAVIGATION: For action configuration, see ActionsCfg below
```

### 5. Progressive Disclosure Structure

```
CLAUDE.md (root)
├── Points to detailed docs
├── Quick command reference
└── Architecture summary

.claude/docs/
├── REPOSITORY_STRUCTURE.md (this file)
├── AGENT_STRATEGY.md
├── RESEARCH_OVERVIEW.md
└── QUICK_REFERENCE.md

.claude/skills/
├── creating-environments/
├── training-policies/
└── debugging-simulation/
```

---

## Appendix: Key Classes Reference

### Environment Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `ManagerBasedEnv` | `isaaclab/envs/manager_based_env.py` | Base environment |
| `ManagerBasedRLEnv` | `isaaclab/envs/manager_based_rl_env.py` | RL environment |
| `DirectRLEnv` | `isaaclab/envs/direct_rl_env.py` | Direct control env |

### Asset Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `Articulation` | `isaaclab/assets/articulation/` | Jointed robots |
| `RigidObject` | `isaaclab/assets/rigid_object/` | Single rigid bodies |
| `DeformableObject` | `isaaclab/assets/deformable_object/` | Soft objects |

### Manager Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `ActionManager` | `isaaclab/managers/action_manager.py` | Action processing |
| `ObservationManager` | `isaaclab/managers/observation_manager.py` | Observation generation |
| `RewardManager` | `isaaclab/managers/reward_manager.py` | Reward computation |
| `TerminationManager` | `isaaclab/managers/termination_manager.py` | Episode termination |
| `EventManager` | `isaaclab/managers/event_manager.py` | Reset/randomization |
| `CommandManager` | `isaaclab/managers/command_manager.py` | Goal generation |

### Controller Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `DifferentialIKController` | `isaaclab/controllers/differential_ik.py` | IK control |
| `OperationalSpaceController` | `isaaclab/controllers/operational_space.py` | OSC control |
| `JointImpedanceController` | `isaaclab/controllers/joint_impedance.py` | Impedance control |

---

*This document serves as a navigation guide for AI agents working with Isaac Lab. For the latest information, always refer to the official documentation at https://isaac-sim.github.io/IsaacLab/*
