---
name: creating-environments
description: Step-by-step guide for creating new RL/IL environments in Isaac Lab. Use when user wants to create a new task, environment, or scenario for robot learning, or needs help designing observations, actions, rewards, or terminations.
---

# Creating Isaac Lab Environments

## Overview

Isaac Lab uses a **manager-based architecture** where MDP components are modular and configurable. This guide walks through creating a new environment.

## Step 1: Choose Your Base

**Locomotion tasks:**
- Base: `/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/`
- Copy the entire `velocity/` folder structure

**Manipulation tasks:**
- Base: `/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/`
- Choose `reach/`, `lift/`, `pick_place/` based on complexity

**Classic control:**
- Base: `/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/`
- Simple environments like cartpole, humanoid

## Step 2: Directory Structure

Create this structure for a new task:

```
my_task/
├── __init__.py                 # Gym registration
├── my_task_env_cfg.py          # Base environment config
├── config/
│   └── my_robot/
│       ├── __init__.py         # Robot-specific registration
│       ├── env_cfg.py          # Robot-specific config overrides
│       └── agents/
│           ├── rsl_rl_ppo_cfg.py    # RSL-RL training config
│           └── skrl_ppo_cfg.yaml    # SKRL training config
└── mdp/                        # Custom MDP components
    ├── __init__.py
    ├── observations.py
    ├── rewards.py
    └── terminations.py
```

## Step 3: Environment Configuration

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for my custom task."""

    # Scene configuration
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP configuration
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Episode settings
    episode_length_s = 20.0
    decimation = 4  # Control frequency = sim_freq / decimation
```

## Step 4: Scene Configuration

```python
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene with robot and environment."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Robot
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Optional: objects, terrain, sensors
```

## Step 5: Observations

```python
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.envs.mdp import observations as obs_mdp

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy network."""

        # Built-in observation terms
        joint_pos = ObservationTermCfg(func=obs_mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=obs_mdp.joint_vel_rel)

        # Custom observation (define in mdp/observations.py)
        custom_obs = ObservationTermCfg(
            func=custom_observation_function,
            params={"scale": 1.0}
        )

    policy: PolicyCfg = PolicyCfg()
```

## Step 6: Actions

```python
from isaaclab.managers import ActionGroupCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg

@configclass
class ActionsCfg(ActionGroupCfg):
    """Action configuration."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # Regex for all joints
        scale=0.5,
        use_default_offset=True,
    )
```

## Step 7: Rewards

```python
from isaaclab.managers import RewardTermCfg
from isaaclab.envs.mdp import rewards as rew_mdp

@configclass
class RewardsCfg:
    """Reward terms."""

    # Built-in rewards
    alive = RewardTermCfg(func=rew_mdp.is_alive, weight=1.0)

    # Custom reward (define in mdp/rewards.py)
    task_success = RewardTermCfg(
        func=custom_reward_function,
        weight=10.0,
        params={"threshold": 0.1}
    )

    # Penalties
    action_rate = RewardTermCfg(func=rew_mdp.action_rate_l2, weight=-0.01)
```

## Step 8: Terminations

```python
from isaaclab.managers import TerminationTermCfg
from isaaclab.envs.mdp import terminations as term_mdp

@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = TerminationTermCfg(func=term_mdp.time_out, time_out=True)

    # Custom termination
    fell_over = TerminationTermCfg(
        func=custom_termination_function,
        params={"height_threshold": 0.3}
    )
```

## Step 9: Gym Registration

In `__init__.py`:

```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-MyTask-MyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:MyTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyTaskPPORunnerCfg",
    },
)
```

## Step 10: Training Configuration

In `agents/rsl_rl_ppo_cfg.py`:

```python
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

@configclass
class MyTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
```

## Step 11: Test Your Environment

```bash
# Test with random actions
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-MyTask-MyRobot-v0

# Train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-MyTask-MyRobot-v0
```

## Custom MDP Functions Template

```python
# mdp/rewards.py
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def custom_reward_function(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """Custom reward computation."""
    asset = env.scene[asset_cfg.name]
    # Your reward logic here
    reward = torch.zeros(env.num_envs, device=env.device)
    return reward
```

## Checklist

- [ ] Created directory structure
- [ ] Defined SceneCfg with robot and objects
- [ ] Configured observations (what agent sees)
- [ ] Configured actions (what agent can do)
- [ ] Designed reward function (what agent optimizes)
- [ ] Set termination conditions
- [ ] Registered with gymnasium
- [ ] Created training config
- [ ] Tested with random agent
- [ ] Verified physics behavior visually
