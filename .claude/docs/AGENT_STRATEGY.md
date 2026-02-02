# Isaac Lab Assistant Agent Strategy

> A comprehensive strategy for building a Claude Code-based agent that helps users create imitation learning and reinforcement learning scenarios in Isaac Lab.

---

## Table of Contents

1. [Vision & Objectives](#vision--objectives)
2. [Agent Architecture](#agent-architecture)
3. [Core Capabilities](#core-capabilities)
4. [Skills System Design](#skills-system-design)
5. [Workflow Patterns](#workflow-patterns)
6. [MuJoCo Translation Layer](#mujoco-translation-layer)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Knowledge Sources](#knowledge-sources)
9. [Example Interactions](#example-interactions)

---

## Vision & Objectives

### Vision

Create an AI assistant that enables any developer to effectively use Isaac Lab for robot learning, regardless of their prior experience with the framework, by providing:

1. **Guided Environment Creation** - Step-by-step assistance creating custom RL/IL environments
2. **MuJoCo Translation** - Help users translate MuJoCo experience to Isaac Lab patterns
3. **Best Practices Enforcement** - Automatic application of Isaac Lab conventions
4. **Debugging Support** - Intelligent troubleshooting of common issues
5. **Training Pipeline Assistance** - End-to-end help from environment to trained policy

### Success Criteria

| Metric | Target |
|--------|--------|
| Environment creation time | 75% reduction for new users |
| Code correctness | 95%+ first-attempt success rate |
| User questions resolved | 90%+ without external documentation |
| MuJoCo translation accuracy | 85%+ functional equivalence |

---

## Agent Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interaction Layer                    │
│  (CLI conversations, natural language task descriptions)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Intent Recognition                        │
│  - Environment creation                                      │
│  - Robot configuration                                       │
│  - MDP component design                                      │
│  - Training setup                                            │
│  - Debugging assistance                                      │
│  - MuJoCo translation                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Skills Dispatcher                         │
│  Routes to appropriate skill based on intent                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Environment   │   │   Training    │   │   Debugging   │
│   Creation    │   │    Setup      │   │   Assistant   │
│    Skill      │   │    Skill      │   │     Skill     │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Code Generation Engine                    │
│  - Template-based generation                                 │
│  - Context-aware modifications                               │
│  - Validation checks                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Isaac Lab Codebase                        │
│  - Environment files                                         │
│  - Configuration files                                       │
│  - Training scripts                                          │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| Intent Recognition | Parse user requests, identify required workflow |
| Skills Dispatcher | Select and orchestrate appropriate skills |
| Code Generation Engine | Generate valid Isaac Lab code from templates |
| Validation Layer | Verify generated code follows conventions |
| Knowledge Base | Store patterns, examples, and common solutions |

---

## Core Capabilities

### 1. Environment Creation Assistant

**Capability:** Guide users through creating new RL/IL environments

**Workflow:**
1. Gather requirements (task type, robot, terrain, objectives)
2. Select appropriate base configuration
3. Generate environment scaffold
4. Implement custom MDP components
5. Register environment with Gymnasium
6. Create training configuration
7. Validate and test

**Key Questions to Ask:**
- What type of task? (locomotion, manipulation, navigation)
- Which robot? (existing or new)
- What observations does the agent need?
- What actions can the agent take?
- What defines success/failure?
- Any curriculum learning needed?

### 2. Reward Function Designer

**Capability:** Help users design effective reward functions

**Approach:**
1. Understand task objectives in natural language
2. Propose reward structure (sparse vs dense)
3. Generate reward function code
4. Suggest weights and normalization
5. Identify potential reward hacking issues

**Reward Design Patterns:**
```python
# Pattern 1: Progress-based reward
reward = current_distance - previous_distance  # Negative = closer

# Pattern 2: Success bonus with shaping
reward = shaping_reward + success_bonus * is_success

# Pattern 3: Multi-objective weighted sum
reward = (
    w1 * tracking_reward +
    w2 * energy_penalty +
    w3 * safety_constraint
)
```

### 3. Robot Configuration Helper

**Capability:** Assist with integrating new robots or modifying existing ones

**Workflow:**
1. Identify robot source (URDF, MJCF, USD)
2. Convert to USD format if needed
3. Generate ArticulationCfg
4. Configure actuators
5. Set up sensors
6. Create asset registration

### 4. Training Pipeline Orchestrator

**Capability:** Set up and monitor training runs

**Features:**
- Select appropriate RL algorithm
- Configure hyperparameters
- Set up logging and checkpointing
- Generate training launch commands
- Interpret training logs

### 5. MuJoCo-to-Isaac Translation

**Capability:** Help users translate MuJoCo knowledge to Isaac Lab

**Translation Mappings:**
| MuJoCo Concept | Isaac Lab Equivalent |
|----------------|---------------------|
| `gym.Env` | `ManagerBasedRLEnv` |
| `_get_obs()` | `ObservationManager` + `ObsTerm` |
| `step(action)` | `ActionManager` + `ActionTerm` |
| `_get_reward()` | `RewardManager` + `RewTerm` |
| `reset()` | `EventManager` + reset mode |
| `model.opt.timestep` | `SimulationCfg.dt` |
| `mujoco.mj_step()` | Automatic in env loop |

### 6. Debugging Assistant

**Capability:** Help diagnose and fix common issues

**Common Issues Database:**
- Simulation instability (exploding physics)
- Reward not improving
- Observations contain NaN
- Robot falls through ground
- Actions not affecting robot
- Training crashes with OOM

---

## Skills System Design

### Skill 1: Creating Environments

**File:** `.claude/skills/creating-environments/SKILL.md`

```yaml
---
name: creating-environments
description: Step-by-step guide for creating new RL/IL environments in Isaac Lab. Use when user wants to create a new task, environment, or scenario for robot learning.
---
```

**Content Structure:**
1. Pre-flight checklist
2. Decision tree for environment type
3. Template generation steps
4. MDP component implementation
5. Testing and validation

### Skill 2: Training Policies

**File:** `.claude/skills/training-policies/SKILL.md`

```yaml
---
name: training-policies
description: Guide for training RL policies in Isaac Lab. Use when user wants to train, fine-tune, or evaluate policies using RSL-RL, SKRL, RL Games, or Stable Baselines 3.
---
```

**Content Structure:**
1. Framework selection guide
2. Hyperparameter recommendations
3. Training launch commands
4. Monitoring and logging
5. Evaluation and deployment

### Skill 3: Adding Robots

**File:** `.claude/skills/adding-robots/SKILL.md`

```yaml
---
name: adding-robots
description: Guide for adding new robots to Isaac Lab. Use when user wants to import URDF/MJCF, configure actuators, or create robot configurations.
---
```

### Skill 4: MuJoCo Translation

**File:** `.claude/skills/mujoco-translation/SKILL.md`

```yaml
---
name: mujoco-translation
description: Translate MuJoCo/Gymnasium environments to Isaac Lab. Use when user mentions MuJoCo, Gymnasium, or wants to port existing environments.
---
```

### Skill 5: Debugging Simulation

**File:** `.claude/skills/debugging-simulation/SKILL.md`

```yaml
---
name: debugging-simulation
description: Diagnose and fix common Isaac Lab issues. Use when user reports errors, unexpected behavior, or training problems.
---
```

### Skill 6: Imitation Learning

**File:** `.claude/skills/imitation-learning/SKILL.md`

```yaml
---
name: imitation-learning
description: Guide for setting up imitation learning with robomimic or IsaacLab-Mimic. Use when user wants to learn from demonstrations.
---
```

---

## Workflow Patterns

### Pattern 1: New Locomotion Environment

```
User: "I want to train a quadruped robot to walk on rough terrain"

Agent Workflow:
1. [IDENTIFY] Task type: locomotion, velocity tracking
2. [SELECT] Base: manager_based/locomotion/velocity/velocity_env_cfg.py
3. [CONFIGURE] Robot: Check available quadrupeds (anymal_c, anymal_d, spot, go1, go2)
4. [CUSTOMIZE] Terrain: Configure rough terrain generator
5. [ADJUST] Rewards: Walking speed, stability, energy efficiency
6. [GENERATE] Files:
   - config/my_robot/__init__.py
   - config/my_robot/rough_env_cfg.py
   - config/my_robot/agents/rsl_rl_ppo_cfg.py
7. [TEST] Run with random actions to verify
8. [TRAIN] Launch training with recommended hyperparameters
```

### Pattern 2: New Manipulation Environment

```
User: "I want a Franka arm to pick and place objects"

Agent Workflow:
1. [IDENTIFY] Task type: manipulation, pick-place
2. [SELECT] Base: manager_based/manipulation/pick_place/
3. [CONFIGURE] Robot: Franka Emika Panda
4. [DEFINE] Objects: Specify object types, sizes
5. [DESIGN] Observations: End-effector pose, object pose, gripper state
6. [DESIGN] Actions: Delta EE position, gripper command
7. [DESIGN] Rewards:
   - Reaching reward
   - Grasping reward
   - Lifting reward
   - Placement reward
8. [GENERATE] Environment configuration
9. [TEST] Verify grasp physics
10. [TRAIN] With appropriate manipulation hyperparameters
```

### Pattern 3: MuJoCo Environment Port

```
User: "I have a MuJoCo environment for a custom robot, how do I port it?"

Agent Workflow:
1. [ANALYZE] Existing MuJoCo environment structure
2. [CONVERT] MJCF to USD using scripts/tools/convert_mjcf.py
3. [MAP] Observation function → ObservationsCfg
4. [MAP] Reward function → RewardsCfg
5. [MAP] Action space → ActionsCfg
6. [MAP] Reset function → EventsCfg
7. [VERIFY] Physics parameters match
8. [TEST] Compare behaviors side-by-side
9. [OPTIMIZE] Leverage GPU parallelization
```

---

## MuJoCo Translation Layer

### Conceptual Mapping

```
┌─────────────────────────────────────────────────────────────┐
│                    MuJoCo/Gymnasium                          │
├─────────────────────────────────────────────────────────────┤
│  class MyEnv(gym.Env):                                      │
│      def __init__(self):                                    │
│          self.model = mujoco.MjModel.from_xml_path(...)     │
│          self.data = mujoco.MjData(self.model)              │
│                                                              │
│      def step(self, action):                                │
│          self.data.ctrl[:] = action                         │
│          mujoco.mj_step(self.model, self.data)              │
│          obs = self._get_obs()                              │
│          reward = self._get_reward()                        │
│          done = self._check_termination()                   │
│          return obs, reward, done, False, {}                │
│                                                              │
│      def reset(self):                                       │
│          mujoco.mj_resetData(self.model, self.data)         │
│          return self._get_obs(), {}                         │
└─────────────────────────────────────────────────────────────┘
                              │
                    Translation Layer
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Isaac Lab                              │
├─────────────────────────────────────────────────────────────┤
│  @configclass                                               │
│  class MyEnvCfg(ManagerBasedRLEnvCfg):                     │
│      scene: SceneCfg = SceneCfg()      # Robot + objects   │
│      actions: ActionsCfg = ActionsCfg() # ctrl → managers  │
│      observations: ObsCfg = ObsCfg()    # _get_obs()       │
│      rewards: RewardsCfg = RewardsCfg() # _get_reward()    │
│      terminations: TermCfg = TermCfg()  # _check_term()    │
│      events: EventsCfg = EventsCfg()    # reset()          │
│                                                              │
│  # Instantiation:                                           │
│  env = gym.make("Isaac-MyEnv-v0")                          │
│  obs, info = env.reset()                                   │
│  obs, reward, terminated, truncated, info = env.step(act)  │
└─────────────────────────────────────────────────────────────┘
```

### Code Translation Examples

**MuJoCo Observation:**
```python
# MuJoCo
def _get_obs(self):
    return np.concatenate([
        self.data.qpos[2:],   # Skip x,y
        self.data.qvel,
        self.data.cinert.flat,
    ])
```

**Isaac Lab Observation:**
```python
# Isaac Lab
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        body_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        body_ang_vel = ObsTerm(func=mdp.base_ang_vel)

    policy: PolicyCfg = PolicyCfg()
```

**MuJoCo Reward:**
```python
# MuJoCo
def _get_reward(self):
    forward_reward = self.data.qvel[0]  # x velocity
    ctrl_cost = 0.1 * np.sum(np.square(self.data.ctrl))
    return forward_reward - ctrl_cost
```

**Isaac Lab Reward:**
```python
# Isaac Lab
@configclass
class RewardsCfg:
    forward_vel = RewTerm(
        func=mdp.base_lin_vel,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1
    )
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Create CLAUDE.md at repository root
- [ ] Implement basic skills structure
- [ ] Create `creating-environments` skill
- [ ] Create `training-policies` skill
- [ ] Test with simple locomotion task

### Phase 2: Core Capabilities (Week 3-4)

- [ ] Implement `adding-robots` skill
- [ ] Implement `mujoco-translation` skill
- [ ] Create template library
- [ ] Build code generation engine
- [ ] Add validation checks

### Phase 3: Advanced Features (Week 5-6)

- [ ] Implement `debugging-simulation` skill
- [ ] Implement `imitation-learning` skill
- [ ] Create workflow orchestration
- [ ] Add training monitoring assistance
- [ ] Build common issues database

### Phase 4: Polish & Iteration (Week 7-8)

- [ ] User testing and feedback
- [ ] Documentation refinement
- [ ] Performance optimization
- [ ] Edge case handling
- [ ] Community contribution guide

---

## Knowledge Sources

### Primary Sources (Embedded in Skills)

1. **Isaac Lab Documentation** - https://isaac-sim.github.io/IsaacLab/
2. **Tutorial Scripts** - `/scripts/tutorials/`
3. **Example Environments** - `/source/isaaclab_tasks/`
4. **Robot Configurations** - `/source/isaaclab_assets/`

### Secondary Sources (Web-Accessible)

1. **MuJoCo Documentation** - https://mujoco.readthedocs.io/
2. **Gymnasium Documentation** - https://gymnasium.farama.org/
3. **RSL-RL Documentation** - https://github.com/leggedrobotics/rsl_rl
4. **SKRL Documentation** - https://skrl.readthedocs.io/

### Research References

1. **EUREKA** - LLM-based reward design
2. **Code as Policies** - LLM code generation for robotics
3. **Language to Rewards** - Natural language to reward functions

---

## Example Interactions

### Example 1: New Locomotion Task

```
User: I want to create an environment where an ANYmal C robot
      learns to walk while carrying a payload on its back.

Agent: I'll help you create this environment. Let me break this down:

**Task Analysis:**
- Base task: Velocity-tracking locomotion
- Robot: ANYmal C (already available)
- Modification: Add payload mass, adjust rewards

**Files I'll create:**
1. `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c_payload/`
   - `__init__.py` - Environment registration
   - `payload_env_cfg.py` - Environment config with payload
   - `agents/rsl_rl_ppo_cfg.py` - Training config

**Key modifications:**
1. Scene: Add payload rigid body attached to robot trunk
2. Events: Randomize payload mass (5-15 kg range)
3. Rewards: Add stability penalty (trunk orientation)
4. Observations: Add payload mass to observation (for adaptation)

Shall I proceed with generating these files?
```

### Example 2: MuJoCo Translation

```
User: I have this MuJoCo environment for a custom gripper:
      [shows code snippet]

Agent: I see your MuJoCo environment. Let me translate the key components:

**Your MuJoCo Environment:**
- Action space: 4D (finger joint positions)
- Observation: Joint positions + object position
- Reward: Distance to object + grasp success
- Termination: Object lifted above threshold

**Isaac Lab Translation:**

```python
@configclass
class GripperEnvCfg(ManagerBasedRLEnvCfg):
    # Your mujoco.MjModel → USD asset + ArticulationCfg
    scene: SceneCfg = SceneCfg(
        gripper=GRIPPER_CFG,
        object=OBJECT_CFG,
    )

    # Your action space
    actions: ActionsCfg = ActionsCfg(
        joint_pos=JointPositionActionCfg(
            asset_name="gripper",
            joint_names=["finger_.*"],
        )
    )

    # Your _get_obs()
    observations: ObservationsCfg = ObservationsCfg(...)

    # Your _get_reward()
    rewards: RewardsCfg = RewardsCfg(...)
```

**Next steps:**
1. Convert your gripper MJCF to USD
2. I'll generate the full configuration
3. We'll test to verify equivalent behavior

Would you like me to start with the MJCF conversion?
```

### Example 3: Debugging Help

```
User: My training isn't improving. Reward stays around -50
      for 500 iterations. Using anymal locomotion.

Agent: Let me help debug this. First, some diagnostic questions:

**Quick Checks:**
1. Is the robot falling/unstable? (physics issue)
2. Are actions reaching the robot? (action scaling issue)
3. Is the reward function computing correctly? (code bug)

**Let me check your configuration:**

[Reads environment config]

**Issues I found:**
1. ❌ Action scale is 0.5 but joint range is ±1.0
   → Robot can only use 50% of its range

2. ⚠️ Reward weights: forward_vel=2.0, energy=-5.0
   → Energy penalty dominates, robot learns to not move

**Recommended fixes:**
```python
# Fix 1: Increase action scale
actions.joint_pos.scale = 1.0

# Fix 2: Rebalance rewards
rewards.forward_vel.weight = 5.0
rewards.energy.weight = -0.5
```

Shall I apply these changes and restart training?
```

---

## Conclusion

This agent strategy provides a comprehensive framework for building an AI assistant that makes Isaac Lab accessible to all developers. By combining:

1. **Structured Skills** - Domain-specific knowledge in modular, loadable units
2. **Workflow Patterns** - Proven sequences for common tasks
3. **Translation Layers** - Bridges from familiar frameworks (MuJoCo)
4. **Debugging Intelligence** - Rapid diagnosis of common issues

The resulting agent will significantly reduce the learning curve for Isaac Lab while maintaining the flexibility and power that experienced users require.

---

*This strategy document should be updated as the agent evolves and new patterns emerge from user interactions.*
