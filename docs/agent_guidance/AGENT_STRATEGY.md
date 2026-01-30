# Claude Code Agent Strategy for Isaac Lab

> **Objective**: Design a comprehensive strategy for creating a Claude Code-based AI assistant that helps developers create reinforcement learning and imitation learning scenarios in Isaac Lab.

---

## Table of Contents
1. [Vision and Goals](#vision-and-goals)
2. [Agent Architecture](#agent-architecture)
3. [Knowledge Integration](#knowledge-integration)
4. [Core Capabilities](#core-capabilities)
5. [Interaction Patterns](#interaction-patterns)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Context File Design](#context-file-design)
8. [Example Workflows](#example-workflows)

---

## Vision and Goals

### Vision Statement

Create an AI-powered development companion that enables any developer—regardless of robotics expertise—to successfully build, train, and deploy RL/IL policies using Isaac Lab, by providing:

1. **Guided Environment Creation**: Step-by-step assistance for creating custom environments
2. **Intelligent Debugging**: Automatic diagnosis of common issues
3. **Cross-Domain Translation**: Bridge knowledge from MuJoCo and other frameworks
4. **Best Practice Enforcement**: Ensure configurations follow Isaac Lab conventions
5. **End-to-End Support**: From concept to trained policy to deployment

### Success Metrics

- User can create a working custom environment in < 30 minutes
- Common errors diagnosed and resolved in < 3 interactions
- MuJoCo users successfully transition to Isaac Lab concepts
- Training pipelines set up correctly on first attempt

---

## Agent Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                   CLAUDE CODE AGENT                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Context    │  │   Code      │  │  Debugging  │         │
│  │  Manager    │  │  Generator  │  │   Engine    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│  ┌──────▼────────────────▼────────────────▼──────┐         │
│  │              Knowledge Base                    │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │         │
│  │  │ Isaac   │ │ MuJoCo  │ │ Common  │         │         │
│  │  │ Lab API │ │ Mapping │ │ Errors  │         │         │
│  │  └─────────┘ └─────────┘ └─────────┘         │         │
│  └───────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │         Isaac Lab             │
              │   (Repository + Runtime)      │
              └───────────────────────────────┘
```

### Core Components

#### 1. Context Manager
Maintains understanding of:
- Current project state
- User's skill level
- Active task/environment being developed
- Previous interactions and decisions

#### 2. Code Generator
Produces:
- Environment configurations
- Reward functions
- Custom MDP components
- Training scripts
- Deployment code

#### 3. Debugging Engine
Handles:
- Error message interpretation
- Physics instability diagnosis
- Training convergence issues
- Configuration validation

#### 4. Knowledge Base
Contains:
- Isaac Lab API documentation
- MuJoCo to Isaac Lab translation rules
- Common error patterns and solutions
- Best practices and templates

---

## Knowledge Integration

### Domain Knowledge Layers

```
┌─────────────────────────────────────────────┐
│ Layer 4: Task-Specific Knowledge            │
│ - Locomotion patterns (gaits, velocities)   │
│ - Manipulation primitives (reach, grasp)    │
│ - Navigation strategies                     │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│ Layer 3: RL/IL Concepts                     │
│ - Reward shaping strategies                 │
│ - Curriculum learning                       │
│ - Domain randomization                      │
│ - Sim-to-real transfer                      │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│ Layer 2: Isaac Lab Specifics                │
│ - Manager system architecture               │
│ - Configuration patterns                    │
│ - Asset management                          │
│ - Physics settings                          │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│ Layer 1: Foundation                         │
│ - PyTorch tensor operations                 │
│ - Gymnasium interface                       │
│ - USD scene description                     │
│ - PhysX physics concepts                    │
└─────────────────────────────────────────────┘
```

### MuJoCo to Isaac Lab Translation

The agent should understand these mappings:

| MuJoCo Concept | Isaac Lab Equivalent | Notes |
|----------------|---------------------|-------|
| `mj_data.qpos` | `articulation.data.joint_pos` | Joint positions |
| `mj_data.qvel` | `articulation.data.joint_vel` | Joint velocities |
| MJCF XML | USD + `ArticulationCfg` | Scene description |
| `mj_step()` | `env.step()` | Simulation step |
| `gym.make()` | `gym.make()` | Same interface! |
| `env.render()` | Omniverse viewport | Different rendering |
| Contact forces | `ContactSensor` | Managed by sensors |
| Domain randomization | `EventManager` | Configuration-driven |

### Skill-Based Knowledge

Understanding hierarchical RL and skills:

```python
# Skills in Isaac Lab context
class SkillConcepts:
    """Agent should understand these skill patterns"""

    # 1. Options Framework mapping
    # - Initiation → Can start conditions in termination manager
    # - Policy → Manager-based or direct env logic
    # - Termination → TerminationManager conditions

    # 2. Skill composition patterns
    # - Sequential skills via curriculum
    # - Parallel skills via multi-task configs
    # - Hierarchical via command managers

    # 3. Imitation learning skills
    # - RecorderManager for demonstrations
    # - SkillGen for automated generation
    # - Behavior cloning pipelines
```

---

## Core Capabilities

### Capability 1: Environment Creation

**User Intent**: "I want to create a reaching task for a Franka arm"

**Agent Response Flow**:
1. Identify task type: Manipulation → Reaching
2. Identify robot: Franka Emika
3. Propose workflow: Manager-based (recommended for prototyping)
4. Generate scaffolding:
   - Scene configuration with Franka
   - Target object spawning
   - Observation manager (joint states, target position)
   - Action manager (joint position control)
   - Reward manager (distance to target)
   - Termination manager (success, timeout)
5. Provide training command

**Generated Code Example**:
```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

@configclass
class FrankaReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Franka reaching task."""

    # Scene setup
    scene: FrankaReachSceneCfg = FrankaReachSceneCfg()

    # Observations: joint positions, velocities, target position
    observations: ObservationsCfg = ObservationsCfg()

    # Actions: 7-DOF joint position commands
    actions: ActionsCfg = ActionsCfg()

    # Rewards: distance to target, action penalty
    rewards: RewardsCfg = RewardsCfg()

    # Terminations: success (close to target), timeout
    terminations: TerminationsCfg = TerminationsCfg()
```

### Capability 2: Reward Function Design

**User Intent**: "The robot isn't learning to walk properly"

**Agent Diagnosis Flow**:
1. Check current reward configuration
2. Identify missing reward terms:
   - Base velocity tracking
   - Base height maintenance
   - Orientation stability
   - Energy efficiency
   - Gait symmetry
3. Propose reward shaping improvements
4. Suggest curriculum learning if needed

**Reward Design Principles**:
```python
# Agent should recommend these patterns:

# 1. Primary task reward (sparse → dense)
velocity_tracking = RewardTermCfg(
    func=mdp.velocity_tracking,
    weight=2.0,  # Highest weight for main objective
)

# 2. Regularization rewards
action_smoothness = RewardTermCfg(
    func=mdp.action_rate,
    weight=-0.01,  # Small negative for penalty
)

# 3. Safety constraints
joint_limit_penalty = RewardTermCfg(
    func=mdp.joint_limits,
    weight=-1.0,  # Strong penalty for violations
)

# 4. Shaping rewards (help learning, reduce at convergence)
base_height = RewardTermCfg(
    func=mdp.base_height,
    weight=0.5,
    params={"target_height": 0.5},
)
```

### Capability 3: Debugging and Diagnostics

**Common Issues and Solutions**:

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| Robot "explodes" | Collision mesh overlap, joint limits exceeded | Check collision geometry, add joint limit penalties |
| NaN in observations | Physics instability, sensor misconfiguration | Reduce timestep, check sensor configs |
| OOM errors | Too many environments | Reduce `num_envs`, increase GPU buffers |
| Training not converging | Reward sparsity, wrong hyperparameters | Add shaping rewards, adjust learning rate |
| Policy oscillates | Action space mismatch | Check action scaling, add action smoothness penalty |

**Diagnostic Workflow**:
```
1. Error Recognition
   └── Parse error message
   └── Identify error category

2. Context Gathering
   └── Check relevant config files
   └── Review recent changes

3. Hypothesis Formation
   └── Most likely causes
   └── Less likely but possible

4. Solution Proposal
   └── Immediate fix
   └── Long-term improvement
   └── Prevention strategy
```

### Capability 4: Training Pipeline Setup

**Full Pipeline Assistance**:

```python
# Agent helps set up complete training workflow

# Step 1: Environment verification
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task My-Custom-Env-v0 --num_envs 16

# Step 2: Training configuration
# Agent generates appropriate agent config based on task type

# Step 3: Training launch
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task My-Custom-Env-v0 \
    --num_envs 4096 \
    --max_iterations 1000

# Step 4: Monitoring
tensorboard --logdir logs/rsl_rl/My-Custom-Env-v0

# Step 5: Evaluation
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task My-Custom-Env-v0 \
    --num_envs 32 \
    --load_run <timestamp>
```

### Capability 5: Sim-to-Real Guidance

**Deployment Assistance**:

1. **Observation Normalization**:
   - Extract running mean/std from training
   - Apply during inference

2. **Action Space Matching**:
   - Verify action scaling matches real robot
   - Account for motor dynamics

3. **Sensor Calibration**:
   - Map simulated sensors to real sensors
   - Handle noise differences

4. **Safety Layers**:
   - Add joint limit enforcement
   - Emergency stop conditions

---

## Interaction Patterns

### Pattern 1: Guided Creation

```
User: I want to train a quadruped to walk

Agent: I'll help you set up a quadruped locomotion task. Let me ask a few questions:

1. Which robot? (Options: ANYmal C, ANYmal D, Unitree Go2, Unitree A1, Spot)
2. Terrain type? (Options: Flat, Rough, Stairs)
3. Control mode? (Options: Velocity tracking, Position following)

User: Go2, rough terrain, velocity tracking

Agent: Great choice! I'll create a velocity-tracking task for Go2 on rough terrain.

[Generates complete configuration]

Here's your environment setup:
- Location: source/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/
- Training command: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-v0

Shall I explain any part of the configuration?
```

### Pattern 2: Problem Diagnosis

```
User: My training crashes with "CUDA out of memory"

Agent: This is a common issue. Let me diagnose:

Checking your configuration...
- num_envs: 8192
- GPU: (detected)
- Physics engine: PhysX

The issue is likely too many parallel environments for your GPU memory.

Solutions (in order of preference):
1. Reduce num_envs to 4096: `--num_envs 4096`
2. Increase PhysX GPU buffer:
   ```python
   sim: SimulationCfg = SimulationCfg(
       physx=PhysxCfg(
           gpu_heap_capacity=2**26,  # Increase from default
           gpu_temp_buffer_capacity=2**24,
       )
   )
   ```
3. Enable gradient checkpointing in your RL config

Try option 1 first. Would you like me to explain the tradeoffs?
```

### Pattern 3: Concept Translation

```
User: In MuJoCo I used env.sim.data.qpos for joint positions. What's the equivalent?

Agent: In Isaac Lab, joint data is accessed through the articulation asset:

MuJoCo → Isaac Lab mapping:

```python
# MuJoCo
joint_pos = env.sim.data.qpos

# Isaac Lab (in environment class)
joint_pos = self.scene["robot"].data.joint_pos

# In reward/observation functions
def get_joint_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos
```

Key differences:
1. Data is batched: shape is [num_envs, num_joints]
2. It's a PyTorch tensor on GPU
3. Access via scene entity, not raw simulation

Would you like me to show more MuJoCo→Isaac Lab translations?
```

### Pattern 4: Best Practice Coaching

```
User: How should I structure my reward function?

Agent: Great question! Here are Isaac Lab reward design best practices:

1. **Hierarchical Structure**:
   ```
   Total Reward = Primary Task + Regularization + Safety
   ```

2. **Weight Guidelines**:
   - Primary task (what you want): weight 1.0-2.0
   - Shaping rewards (helpers): weight 0.1-0.5
   - Penalties (constraints): weight -0.01 to -1.0

3. **Common Patterns**:
   ```python
   rewards: RewardsCfg = RewardsCfg(
       # Primary
       velocity_tracking=RewardTermCfg(func=mdp.track_velocity, weight=1.5),

       # Shaping (reduce weight as training progresses via curriculum)
       alive=RewardTermCfg(func=mdp.is_alive, weight=0.5),

       # Regularization
       action_rate=RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),

       # Safety
       joint_limits=RewardTermCfg(func=mdp.joint_limits_exceeded, weight=-1.0),
   )
   ```

4. **Curriculum Integration**:
   Start with easier rewards, increase difficulty over time.

Would you like me to analyze your current rewards and suggest improvements?
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

1. **Create CLAUDE.md context file**
   - Repository structure overview
   - Common commands
   - Key patterns

2. **Build pattern templates**
   - Environment scaffolding
   - Reward function templates
   - Training scripts

3. **Document MuJoCo mappings**
   - Concept translation guide
   - Code pattern equivalents

### Phase 2: Core Agent Logic (Week 3-4)

1. **Implement guided creation workflow**
   - Question-answer flow for new environments
   - Template selection and customization
   - Code generation with validation

2. **Build error diagnosis system**
   - Error pattern matching
   - Solution lookup table
   - Contextual debugging

3. **Create knowledge retrieval**
   - API documentation indexing
   - Example code search
   - Best practice database

### Phase 3: Advanced Features (Week 5-6)

1. **Training monitoring integration**
   - Log analysis
   - Convergence detection
   - Automatic suggestions

2. **Sim-to-real guidance**
   - Deployment checklist
   - Calibration helpers
   - Safety verification

3. **Skill/IL support**
   - Demonstration collection guidance
   - SkillGen integration
   - Behavior cloning workflows

### Phase 4: Polish and Testing (Week 7-8)

1. **User testing**
   - Beginner workflow validation
   - Expert workflow optimization
   - Edge case handling

2. **Documentation**
   - User guide
   - Example conversations
   - Troubleshooting guide

3. **Continuous improvement**
   - Feedback collection
   - Pattern updates
   - New feature integration

---

## Context File Design

### Main Context File: `CLAUDE.md`

Create at `/home/user/isaaclab/CLAUDE.md`:

```markdown
# Isaac Lab Development Context

## Repository Overview
Isaac Lab is NVIDIA's GPU-accelerated robot learning framework.

## Quick Commands
```bash
# Train with RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK>

# Evaluate trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task <TASK> --load_run <RUN>

# List available environments
./isaaclab.sh -p scripts/environments/list_envs.py
```

## Project Structure
- `source/isaaclab/` - Core framework
- `source/isaaclab_tasks/` - Environment definitions
- `source/isaaclab_assets/` - Robot configurations
- `scripts/` - Training and demo scripts

## Creating Custom Environments
1. Choose workflow: Manager-based (recommended) or Direct
2. Define configuration using @configclass
3. Implement reward functions in mdp/rewards.py
4. Register environment in __init__.py
5. Create RL agent configs in agents/ directory

## Common Patterns
```python
# Environment configuration
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

# Reward function
def my_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)
```

## Troubleshooting
- OOM: Reduce num_envs or increase GPU buffers
- Robot exploding: Check collision meshes and joint limits
- NaN observations: Review sensor configs and physics stability
- Training not converging: Add shaping rewards, check hyperparameters

## Key Documentation
- Tutorials: docs/source/_static/tutorials/
- API Reference: docs/source/api/
- Examples: Look at manager_based/classic/cartpole/ for simple example

## MuJoCo Users
- qpos → scene["robot"].data.joint_pos
- qvel → scene["robot"].data.joint_vel
- mj_step() → env.step()
- Rewards defined declaratively in config, not inline
```

### Supplementary Context Files

Create topic-specific context files:

1. **`CLAUDE_REWARDS.md`** - Reward design patterns
2. **`CLAUDE_DEBUGGING.md`** - Error diagnosis guide
3. **`CLAUDE_MUJOCO.md`** - MuJoCo translation guide
4. **`CLAUDE_DEPLOYMENT.md`** - Sim-to-real guidance

---

## Example Workflows

### Workflow 1: Complete Environment Creation

```
User: Create a pick-and-place task for a UR10 arm

Agent:
I'll create a complete pick-and-place environment for UR10.

**Step 1: Scene Setup**
```python
@configclass
class UR10PickPlaceSceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10.usd"
        ),
        actuators={"arm": ImplicitActuatorCfg(...)},
    )

    table = AssetBaseCfg(...)
    object = RigidObjectCfg(...)
    target = VisualizationMarkersCfg(...)
```

**Step 2: Observations**
- End-effector position (3D)
- Object position (3D)
- Target position (3D)
- Joint positions (6D)
- Gripper state (1D)
Total: 16-dimensional observation

**Step 3: Actions**
- Delta end-effector position (3D)
- Gripper open/close (1D)
Total: 4-dimensional action

**Step 4: Rewards**
```python
# Reaching phase
reach_object = RewardTermCfg(func=mdp.distance_ee_to_object, weight=1.0)

# Grasp phase
grasp_success = RewardTermCfg(func=mdp.object_grasped, weight=2.0)

# Transport phase
object_to_target = RewardTermCfg(func=mdp.distance_object_to_target, weight=1.5)

# Place phase
place_success = RewardTermCfg(func=mdp.object_at_target, weight=5.0)

# Regularization
action_penalty = RewardTermCfg(func=mdp.action_rate, weight=-0.01)
```

Shall I generate the complete files and show you how to train?
```

### Workflow 2: Debugging Training Issues

```
User: Training loss is NaN after 100 iterations

Agent:
NaN in training typically indicates numerical instability. Let me diagnose:

**Checking common causes:**

1. **Physics Instability** - Most likely cause
   - Check: Are observations reasonable before NaN?
   - Fix: Add observation clipping
   ```python
   observations: ObservationsCfg = ObservationsCfg(
       policy=ObservationGroupCfg(
           enable_corruption=True,
           clip=(-100.0, 100.0),  # Add clipping
       )
   )
   ```

2. **Reward Explosion**
   - Check: What are reward values before crash?
   - Fix: Clip rewards
   ```python
   class MyEnvCfg(ManagerBasedRLEnvCfg):
       clip_rewards = 10.0  # Limit reward magnitude
   ```

3. **Action Space Issues**
   - Check: Are actions properly scaled?
   - Fix: Verify action limits match robot capabilities

4. **Learning Rate Too High**
   - Fix: Reduce learning rate
   ```python
   learning_rate = 1e-4  # Try lower if currently higher
   ```

**Diagnostic command:**
```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Your-Task-v0 --num_envs 4
```
Watch the simulation - does the robot behave reasonably?

Would you like me to examine your specific configuration files?
```

### Workflow 3: MuJoCo Migration

```
User: I have a MuJoCo Ant environment I want to port to Isaac Lab

Agent:
I'll help you migrate your MuJoCo Ant to Isaac Lab. Good news - Isaac Lab already has Ant!

**Quick Option: Use Existing Ant**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-v0 --num_envs 4096
```

**Custom Migration: If you have modifications**

Here's the mapping for key components:

| Your MuJoCo Code | Isaac Lab Equivalent |
|------------------|---------------------|
| `self.sim.data.qpos[:15]` | `self.scene["ant"].data.joint_pos` |
| `self.sim.data.cfrc_ext` | `ContactSensor` readings |
| `self.do_simulation(a, 5)` | Handled by `decimation` in config |

**Your Reward Function Migration:**
```python
# MuJoCo style
def step(self, action):
    forward_reward = self.sim.data.qvel[0]
    survive_reward = 1.0
    ctrl_cost = 0.5 * np.sum(action**2)
    contact_cost = ...

# Isaac Lab style
@configclass
class RewardsCfg:
    forward_velocity = RewardTermCfg(
        func=mdp.root_lin_vel_xy,
        weight=1.0,
    )
    alive = RewardTermCfg(
        func=mdp.is_alive,
        weight=1.0,
    )
    action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.5,
    )
```

**Key Differences to Note:**
1. Vectorized: Everything operates on [num_envs, ...] batches
2. Declarative: Rewards/observations defined in config, not computed inline
3. GPU: All tensors are on GPU (torch.Tensor)

Would you like me to generate a complete custom Ant environment matching your modifications?
```

---

## Summary

This strategy provides a comprehensive framework for building a Claude Code agent that can:

1. **Guide users** through environment creation step-by-step
2. **Generate correct code** following Isaac Lab patterns
3. **Diagnose issues** using knowledge of common problems
4. **Translate concepts** from MuJoCo and other frameworks
5. **Optimize workflows** for both beginners and experts

The key to success is deep integration with the Isaac Lab codebase through context files, pattern templates, and a structured knowledge base that the agent can draw upon during conversations.

---

*This strategy document should be reviewed and updated as Isaac Lab evolves and user patterns emerge.*
