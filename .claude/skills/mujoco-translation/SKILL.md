---
name: mujoco-translation
description: Translate MuJoCo/Gymnasium environments to Isaac Lab. Use when user mentions MuJoCo, Gymnasium, dm_control, wants to port existing environments, or asks about differences between simulators.
---

# MuJoCo to Isaac Lab Translation Guide

## Conceptual Overview

MuJoCo uses a **procedural** approach where you explicitly call physics and build observations. Isaac Lab uses a **manager-based** approach where components are declaratively configured.

```
MuJoCo (Procedural)          →    Isaac Lab (Declarative)
─────────────────────────         ─────────────────────────
class MyEnv(gym.Env):             @configclass
    def step(self, action):       class MyEnvCfg(ManagerBasedRLEnvCfg):
        self.do_simulation()          actions: ActionsCfg
        obs = self._get_obs()         observations: ObservationsCfg
        reward = self._reward()       rewards: RewardsCfg
        return obs, reward, ...       terminations: TerminationsCfg
```

## Core Concept Mapping

| MuJoCo/Gymnasium | Isaac Lab | Notes |
|------------------|-----------|-------|
| `gym.Env` | `ManagerBasedRLEnv` | Base class |
| `mujoco.MjModel` | USD asset + `ArticulationCfg` | Robot definition |
| `mujoco.MjData` | `Articulation` runtime object | State container |
| `env.observation_space` | `ObservationsCfg` | Observation spec |
| `env.action_space` | `ActionsCfg` | Action spec |
| `env._get_obs()` | `ObservationManager` | Observation computation |
| `env.step()` | `ManagerBasedRLEnv.step()` | Automatic |
| `env.reset()` | `EventManager` | Reset handling |
| `model.opt.timestep` | `SimulationCfg.dt` | Physics timestep |
| `frame_skip` | `decimation` | Action repeat |

## Step-by-Step Translation

### Step 1: Convert Robot Model

**MuJoCo MJCF:**
```xml
<mujoco>
  <worldbody>
    <body name="robot">
      <joint name="joint1" type="hinge"/>
      <geom name="link1" type="box"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1" gear="100"/>
  </actuator>
</mujoco>
```

**Convert to USD:**
```bash
./isaaclab.sh -p scripts/tools/convert_mjcf.py \
    robot.xml \
    robot.usd \
    --merge-fixed-joints
```

**Create ArticulationCfg:**
```python
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path="robot.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={"joint1": 0.0},
    ),
    actuators={
        "motor": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            stiffness=0.0,      # For torque control
            damping=0.0,
            effort_limit=100.0,
        ),
    },
)
```

### Step 2: Translate Observations

**MuJoCo:**
```python
def _get_obs(self):
    return np.concatenate([
        self.data.qpos.flat,      # Joint positions
        self.data.qvel.flat,      # Joint velocities
        self.data.cinert.flat,    # Center of mass inertia
    ])
```

**Isaac Lab:**
```python
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs.mdp import observations as mdp

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # qpos equivalent
        joint_pos = ObsTerm(func=mdp.joint_pos)

        # qvel equivalent
        joint_vel = ObsTerm(func=mdp.joint_vel)

        # For body states, use base velocity
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

    policy: PolicyCfg = PolicyCfg()
```

### Step 3: Translate Actions

**MuJoCo:**
```python
def step(self, action):
    self.data.ctrl[:] = action  # Direct actuator control
    mujoco.mj_step(self.model, self.data)
```

**Isaac Lab:**
```python
from isaaclab.envs.mdp.actions import JointEffortActionCfg

@configclass
class ActionsCfg(ActionGroupCfg):
    # For torque control (like MuJoCo ctrl)
    joint_effort = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )

# Or for position control:
from isaaclab.envs.mdp.actions import JointPositionActionCfg

@configclass
class ActionsCfg(ActionGroupCfg):
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )
```

### Step 4: Translate Rewards

**MuJoCo:**
```python
def _get_reward(self):
    forward_reward = self.data.qvel[0]  # x velocity
    ctrl_cost = 0.1 * np.sum(np.square(self.data.ctrl))
    healthy_reward = 1.0 if self._is_healthy() else 0.0
    return forward_reward - ctrl_cost + healthy_reward
```

**Isaac Lab:**
```python
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs.mdp import rewards as mdp

@configclass
class RewardsCfg:
    # forward_reward equivalent
    lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # ctrl_cost equivalent
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # healthy_reward equivalent
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
```

### Step 5: Translate Terminations

**MuJoCo:**
```python
def _is_terminated(self):
    # Height-based termination
    height = self.data.qpos[2]
    return height < 0.3 or height > 2.0
```

**Isaac Lab:**
```python
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp import terminations as mdp

@configclass
class TerminationsCfg:
    # Height termination
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3}
    )

    # Time limit
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

### Step 6: Translate Reset/Randomization

**MuJoCo:**
```python
def reset(self, seed=None):
    mujoco.mj_resetData(self.model, self.data)
    # Add noise
    self.data.qpos += np.random.uniform(-0.1, 0.1, self.model.nq)
    return self._get_obs()
```

**Isaac Lab:**
```python
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.mdp import events as mdp

@configclass
class EventsCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # Add noise to initial state
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        }
    )
```

## Common MuJoCo Patterns

### Pattern: Contact Forces

**MuJoCo:**
```python
contact_forces = self.data.cfrc_ext
```

**Isaac Lab:**
```python
# Use contact sensor
from isaaclab.sensors import ContactSensorCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

# Access in reward/observation
def contact_force_obs(env):
    return env.scene["contact_sensor"].data.net_forces_w
```

### Pattern: Sensor Reading

**MuJoCo:**
```python
# Site position
site_pos = self.data.site_xpos[site_id]
```

**Isaac Lab:**
```python
# Use FrameTransformer
from isaaclab.sensors import FrameTransformerCfg

frame_transformer = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    target_frames=[
        FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/ee_link"),
    ],
)

# Access: env.scene["frame_transformer"].data.target_pos_w
```

### Pattern: Domain Randomization

**MuJoCo:**
```python
def reset(self):
    self.model.body_mass[1] *= np.random.uniform(0.8, 1.2)
    self.model.geom_friction[:, 0] = np.random.uniform(0.5, 1.5)
```

**Isaac Lab:**
```python
@configclass
class EventsCfg:
    # Mass randomization
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        }
    )

    # Friction randomization (in PhysicsMaterialCfg)
```

## Physics Parameter Mapping

| MuJoCo | Isaac Lab | Location |
|--------|-----------|----------|
| `model.opt.timestep` | `SimulationCfg.dt` | Env config |
| `model.opt.gravity` | `SimulationCfg.gravity` | Env config |
| `body_mass` | Asset USD or randomization | Scene/Events |
| `geom_friction` | `PhysicsMaterialCfg` | Asset config |
| `joint_damping` | `ActuatorCfg.damping` | Asset config |
| `joint_frictionloss` | `ActuatorCfg.friction` | Asset config |

## Performance Notes

| Aspect | MuJoCo | Isaac Lab |
|--------|--------|-----------|
| Parallelization | CPU multiprocessing | GPU parallel (CUDA) |
| Typical envs | 8-64 | 2048-8192 |
| Speed (steps/sec) | ~100K | ~1M+ |
| Rendering | Slow | Fast (Omniverse) |

## Verification Checklist

- [ ] Robot spawns in correct pose
- [ ] Joint limits match original
- [ ] Actuator behavior similar (PD gains, torque limits)
- [ ] Observations have same dimensions
- [ ] Action scaling matches expected behavior
- [ ] Rewards compute similar values
- [ ] Episode terminates correctly
- [ ] Physics behavior visually similar
