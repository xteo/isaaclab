# Isaac Lab Repository Structure & Agent Navigation Guide

> **Purpose**: This document provides a comprehensive map of the Isaac Lab repository optimized for AI agents (Claude Code) to navigate and assist developers with reinforcement learning and imitation learning tasks.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Root Directory Structure](#root-directory-structure)
3. [Core Packages Breakdown](#core-packages-breakdown)
4. [Key Architectural Patterns](#key-architectural-patterns)
5. [Entry Points for Common Tasks](#entry-points-for-common-tasks)
6. [Agent Navigation Quick Reference](#agent-navigation-quick-reference)
7. [Instrumentation Recommendations](#instrumentation-recommendations)

---

## Executive Summary

Isaac Lab is NVIDIA's GPU-accelerated robot learning framework built on Isaac Sim. The codebase follows a **modular, configuration-driven architecture** where:

- **Everything is configured via `@configclass` dataclasses**
- **Managers decompose MDP logic** (observations, actions, rewards, terminations)
- **Parallel environments run on GPU** using PyTorch tensors
- **Multiple RL frameworks are supported** through wrapper layers

### Quick Facts
- **Location**: `/home/user/isaaclab/`
- **Core Version**: 0.42.25
- **Task Version**: 0.10.41
- **Physics Engines**: PhysX (default), Newton (experimental)
- **RL Frameworks**: RSL-RL, SKRL, RL Games, Stable-Baselines3

---

## Root Directory Structure

```
/home/user/isaaclab/
├── source/                     # Core source code (7 packages)
│   ├── isaaclab/               # Core framework (envs, managers, assets)
│   ├── isaaclab_tasks/         # Task/environment definitions
│   ├── isaaclab_assets/        # Robot configurations
│   ├── isaaclab_rl/            # RL framework wrappers
│   ├── isaaclab_newton/        # Newton physics integration
│   ├── isaaclab_experimental/  # Experimental features
│   └── isaaclab_tasks_experimental/
├── scripts/                    # Training and demo scripts
│   ├── reinforcement_learning/ # RL training entry points
│   │   ├── rsl_rl/             # RSL-RL (legged locomotion)
│   │   ├── sb3/                # Stable-Baselines3
│   │   ├── rl_games/           # RL Games
│   │   └── skrl/               # SKRL
│   ├── environments/           # Environment demos
│   └── benchmarks/             # Performance benchmarks
├── docs/                       # Sphinx documentation
├── apps/                       # Applications and utilities
├── tools/                      # Development tools
├── docker/                     # Docker configurations
├── pyproject.toml              # Main project configuration
└── isaaclab.sh / isaaclab.bat  # Platform launcher scripts
```

---

## Core Packages Breakdown

### 1. `isaaclab` - Core Framework
**Location**: `/home/user/isaaclab/source/isaaclab/isaaclab/`

```
isaaclab/
├── envs/                       # Environment base classes
│   ├── manager_based_env.py    # Base for manager-based envs
│   ├── manager_based_rl_env.py # RL-specific extension
│   ├── direct_rl_env.py        # Direct RL environment
│   └── *_cfg.py                # Configuration classes
│
├── managers/                   # MDP Component Managers [CRITICAL]
│   ├── observation_manager.py  # Computes observations
│   ├── action_manager.py       # Processes actions
│   ├── reward_manager.py       # Aggregates rewards
│   ├── termination_manager.py  # Episode termination
│   ├── event_manager.py        # Randomization events
│   ├── command_manager.py      # Command generation
│   ├── curriculum_manager.py   # Training curriculum
│   └── recorder_manager.py     # Data recording (for IL)
│
├── scene/                      # Scene management
│   ├── interactive_scene.py    # Main scene class
│   └── interactive_scene_cfg.py
│
├── assets/                     # Asset management
│   ├── articulation/           # Robot articulations
│   └── rigid_object/           # Rigid bodies
│
├── sensors/                    # Sensor implementations
│   ├── camera/                 # RGB/Depth cameras
│   └── contact_sensor/         # Contact forces
│
├── actuators/                  # Motor/actuator models
│   ├── actuator_pd.py          # PD controller
│   └── actuator_net.py         # Neural network actuator
│
├── sim/                        # Simulation backend
│   ├── spawners/               # Asset spawning
│   ├── converters/             # Data converters
│   └── schemas/                # USD schemas
│
├── terrains/                   # Terrain generation
│   ├── height_field/           # Heightmap terrains
│   └── trimesh/                # Mesh terrains
│
├── utils/                      # Utilities
│   ├── noise/                  # Noise generators
│   ├── modifiers/              # Observation modifiers
│   ├── datasets/               # Dataset utilities (IL)
│   └── buffers/                # Tensor buffers
│
└── app/                        # Application launcher
    └── app_launcher.py         # Entry point for sim
```

### 2. `isaaclab_tasks` - Environment Definitions
**Location**: `/home/user/isaaclab/source/isaaclab_tasks/isaaclab_tasks/`

```
isaaclab_tasks/
├── manager_based/              # Manager-based environments [RECOMMENDED]
│   ├── locomotion/
│   │   └── velocity/           # Velocity tracking
│   │       ├── velocity_env_cfg.py
│   │       ├── mdp/            # MDP components
│   │       │   ├── rewards.py
│   │       │   ├── terminations.py
│   │       │   └── curriculums.py
│   │       └── config/
│   │           ├── anymal_c/
│   │           ├── anymal_d/
│   │           ├── h1/         # Unitree H1 humanoid
│   │           ├── spot/       # Boston Dynamics Spot
│   │           └── go2/        # Unitree Go2
│   ├── classic/                # Classic control tasks
│   │   ├── cartpole/
│   │   ├── ant/
│   │   └── humanoid/
│   └── manipulation/           # Manipulation tasks
│       └── reach/
│
├── direct/                     # Direct RL environments
│   ├── locomotion/
│   ├── cartpole/
│   ├── humanoid/
│   ├── allegro_hand/
│   └── inhand_manipulation/
│
└── utils/                      # Task utilities
    ├── hydra.py                # Hydra integration
    └── parse_cfg.py            # Config parsing
```

### 3. `isaaclab_assets` - Robot Definitions
**Location**: `/home/user/isaaclab/source/isaaclab_assets/isaaclab_assets/`

```
isaaclab_assets/
└── robots/
    ├── anymal.py               # ANYmal quadruped
    ├── spot.py                 # Boston Dynamics Spot
    ├── unitree.py              # Unitree robots (A1, Go2, H1, G1)
    ├── franka.py               # Franka Emika arm
    ├── universal_robots.py     # UR5, UR10
    ├── cartpole.py             # Classic cartpole
    ├── humanoid.py             # Humanoid robot
    ├── ant.py                  # Ant quadruped
    └── allegro.py              # Allegro Hand
```

---

## Key Architectural Patterns

### Pattern 1: Configuration via `@configclass`

All Isaac Lab components are configured using dataclass-based configurations:

```python
from isaaclab.utils import configclass

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration"""

    # Scene setup
    scene: MySceneCfg = MySceneCfg(num_envs=4096)

    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
```

**Agent Guidance**: When helping users create environments, always structure configs using `@configclass` and inherit from appropriate base classes.

### Pattern 2: Manager System

The manager system decomposes the MDP into modular components:

| Manager | Purpose | Config Class |
|---------|---------|--------------|
| `ObservationManager` | Compute observations | `ObservationsCfg` |
| `ActionManager` | Process actions | `ActionsCfg` |
| `RewardManager` | Compute rewards | `RewardsCfg` |
| `TerminationManager` | Check termination | `TerminationsCfg` |
| `EventManager` | Handle randomization | `EventCfg` |
| `CommandManager` | Generate commands | `CommandsCfg` |
| `CurriculumManager` | Adjust difficulty | `CurriculumCfg` |
| `RecorderManager` | Record for IL | `RecorderCfg` |

**Agent Guidance**: When debugging, check individual manager configurations. Each manager can be independently tested.

### Pattern 3: MDP Namespace (`mdp/`)

Each task contains an `mdp/` directory with modular reward/termination functions:

```python
# In mdp/rewards.py
def velocity_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for tracking velocity command."""
    command = env.command_manager.get_command("base_velocity")
    actual = env.scene["robot"].data.root_lin_vel_w[:, :2]
    error = torch.sum(torch.square(command - actual), dim=1)
    return torch.exp(-error / 0.25)
```

**Agent Guidance**: Custom rewards go in `mdp/rewards.py`. Reference them in config via function name.

### Pattern 4: Environment Registration

Environments are registered with Gymnasium for discovery:

```python
# In __init__.py
import gymnasium as gym

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents:rsl_rl_ppo_cfg",
    }
)
```

**Agent Guidance**: To find available environments, check `__init__.py` files in task directories or use `scripts/environments/list_envs.py`.

### Pattern 5: Two Environment Workflows

Isaac Lab supports two approaches:

| Workflow | Best For | Base Class |
|----------|----------|------------|
| **Manager-Based** | Rapid prototyping, modularity | `ManagerBasedRLEnv` |
| **Direct** | Performance, custom logic | `DirectRLEnv` |

---

## Entry Points for Common Tasks

### Training RL Policies

```bash
# RSL-RL (recommended for locomotion)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Anymal-C-v0 \
    --num_envs 4096

# Stable-Baselines3
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-Cartpole-v0

# SKRL
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Humanoid-v0
```

**Key Files**:
- Training: `/scripts/reinforcement_learning/<framework>/train.py`
- Inference: `/scripts/reinforcement_learning/<framework>/play.py`

### Testing Environments

```bash
# List all environments
./isaaclab.sh -p scripts/environments/list_envs.py

# Test with random actions
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Cartpole-v0
```

### Creating Custom Environments

1. **Manager-Based** (Recommended):
   - Start from: `/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/`
   - Tutorial: `docs/source/_static/tutorials/03_envs/create_manager_rl_env.html`

2. **Direct**:
   - Start from: `/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/`
   - Tutorial: `docs/source/_static/tutorials/03_envs/create_direct_rl_env.html`

### Adding Custom Robots

1. Define `ArticulationCfg` in `isaaclab_assets/robots/my_robot.py`
2. Export from `isaaclab_assets/__init__.py`
3. Reference in environment config

**Key File**: `/source/isaaclab_assets/isaaclab_assets/robots/`

### Imitation Learning

1. Use `RecorderManager` to capture demonstrations
2. Dataset utilities: `/source/isaaclab/isaaclab/utils/datasets/`
3. SkillGen integration: `docs/source/overview/imitation-learning/skillgen.html`

---

## Agent Navigation Quick Reference

### Finding Things Quickly

| Looking For | Where to Look |
|-------------|---------------|
| Environment configs | `isaaclab_tasks/*/config/` |
| Reward functions | `isaaclab_tasks/*/mdp/rewards.py` |
| Robot definitions | `isaaclab_assets/robots/` |
| Training scripts | `scripts/reinforcement_learning/` |
| Manager implementations | `isaaclab/managers/` |
| Sensor configs | `isaaclab/sensors/` |
| Terrain generation | `isaaclab/terrains/` |
| Actuator models | `isaaclab/actuators/` |

### Common Config Locations by Robot

| Robot | Locomotion Config | RL Agent Config |
|-------|------------------|-----------------|
| ANYmal C | `manager_based/locomotion/velocity/config/anymal_c/` | `anymal_c/agents/` |
| Spot | `manager_based/locomotion/velocity/config/spot/` | `spot/agents/` |
| H1 Humanoid | `manager_based/locomotion/velocity/config/h1/` | `h1/agents/` |
| Go2 | `manager_based/locomotion/velocity/config/go2/` | `go2/agents/` |

### Error Diagnostic Locations

| Error Type | Check These Files |
|------------|------------------|
| Physics instability | Scene config, PhysX settings |
| OOM errors | `num_envs`, GPU buffer configs |
| Observation NaN | Observation manager, sensor configs |
| Reward issues | `mdp/rewards.py`, reward weights |
| Action scaling | Action manager config |

---

## Instrumentation Recommendations

### For AI Agent Navigation

To optimize this repository for Claude Code navigation, implement the following:

#### 1. Create `CLAUDE.md` Context File

Create `/home/user/isaaclab/CLAUDE.md` with:

```markdown
# Isaac Lab Agent Context

## Repository Purpose
GPU-accelerated robot learning framework for RL and IL.

## Key Commands
- Train: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK>`
- Play: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task <TASK>`
- List envs: `./isaaclab.sh -p scripts/environments/list_envs.py`

## Architecture
- Manager-based environments (recommended): `source/isaaclab_tasks/manager_based/`
- Direct environments: `source/isaaclab_tasks/direct/`
- Robot configs: `source/isaaclab_assets/robots/`

## Creating Environments
1. Copy template from `manager_based/classic/cartpole/`
2. Define scene, observations, actions, rewards, terminations
3. Register in `__init__.py`
4. Add RL agent configs in `agents/` subdirectory

## Common Issues
- Robot exploding: Check collision meshes, joint limits
- OOM: Reduce `num_envs`, increase GPU buffer sizes
- NaN observations: Check sensor configs, physics stability
```

#### 2. Add Type Hints and Docstrings

Ensure all public APIs have comprehensive docstrings:

```python
def create_velocity_reward(
    env: ManagerBasedRLEnv,
    target_velocity: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute reward for tracking target velocity.

    Args:
        env: The Isaac Lab environment instance
        target_velocity: Desired forward velocity in m/s
        weight: Reward weight multiplier

    Returns:
        torch.Tensor: Reward values for each environment [num_envs]

    Example:
        >>> reward = create_velocity_reward(env, target_velocity=2.0)
    """
```

#### 3. Create Task Templates

Add template directories for common use cases:

```
templates/
├── locomotion_task/
│   ├── __init__.py.template
│   ├── env_cfg.py.template
│   └── mdp/
│       ├── rewards.py.template
│       └── terminations.py.template
├── manipulation_task/
└── custom_robot/
```

#### 4. Add Searchable Index

Create `/home/user/isaaclab/docs/SEARCHABLE_INDEX.md`:

```markdown
# Searchable Index

## Environments
- Isaac-Velocity-Flat-Anymal-C-v0: Flat terrain locomotion
- Isaac-Velocity-Rough-Anymal-C-v0: Rough terrain locomotion
- Isaac-Cartpole-v0: Classic cartpole balance
...

## Rewards
- velocity_tracking: Track commanded velocity
- base_height: Maintain target height
- orientation_penalty: Penalize non-upright orientations
...

## Terminations
- time_out: Episode time limit
- illegal_contact: Forbidden body contacts
- bad_orientation: Excessive tilt
...
```

#### 5. Implement MCP Server Integration

For real-time agent assistance, create an MCP server:

```
tools/mcp/
├── isaac_lab_mcp_server.py
├── handlers/
│   ├── environment_handler.py
│   ├── config_handler.py
│   └── training_handler.py
└── prompts/
    ├── create_environment.txt
    └── debug_training.txt
```

This would enable:
- Dynamic environment introspection
- Config validation
- Training monitoring
- Error diagnosis

---

## Quick Reference Card

### File Patterns

```
# Environment config
source/isaaclab_tasks/isaaclab_tasks/<workflow>/<domain>/<task>/config/<robot>/*_env_cfg.py

# Reward functions
source/isaaclab_tasks/isaaclab_tasks/<workflow>/<domain>/<task>/mdp/rewards.py

# Agent configs (per framework)
source/isaaclab_tasks/isaaclab_tasks/<workflow>/<domain>/<task>/config/<robot>/agents/*_cfg.py

# Robot definition
source/isaaclab_assets/isaaclab_assets/robots/<robot>.py
```

### Import Patterns

```python
# Core environment
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

# Managers
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg

# Assets
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

# Utils
from isaaclab.utils import configclass
```

### Training Workflow

```
1. Select/Create Environment Config
2. Select RL Framework (RSL-RL, SKRL, SB3, RL-Games)
3. Configure Agent (learning rate, batch size, etc.)
4. Run Training Script
5. Monitor with TensorBoard
6. Evaluate with Play Script
7. Deploy to Real Robot (optional)
```

---

*This document should be updated as Isaac Lab evolves. Last updated based on version 0.42.25.*
