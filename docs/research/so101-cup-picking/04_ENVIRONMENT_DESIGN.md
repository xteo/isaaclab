# Custom Cup-Picking Environment Design

## Overview

This document details how to create a custom Isaac Lab environment for the SO-101 to pick up a small cup. The design is based on the existing **`LiftEnvCfg`** (used by `Isaac-Lift-Cube-OpenArm-v0`) and the **visuomotor stack config** (`stack_ik_rel_visuomotor_env_cfg.py`).

## Reference Implementation

The closest existing environment in Isaac Lab is:
```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/
```

Key files:
- `lift_env_cfg.py` - Base environment config (scene, rewards, observations)
- `mdp/rewards.py` - Reward functions (reaching, lifting, goal tracking)
- `mdp/observations.py` - Observation functions
- `mdp/terminations.py` - Episode termination conditions
- `config/openarm/joint_pos_env_cfg.py` - OpenArm-specific config
- `config/openarm/agents/rsl_rl_ppo_cfg.py` - PPO hyperparameters

## Scene Configuration

### Scene Elements

```python
@configclass
class CupPickingSceneCfg(InteractiveSceneCfg):
    """Scene with SO-101 arm, table, and cup."""

    # SO-101 robot arm (mounted on table)
    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # End-effector frame transformer (for reward computation)
    ee_frame: FrameTransformerCfg = ...  # tracks gripper tip position

    # Target object: small cup (rigid body)
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.2, 0, 0.05],  # in front of the arm, on the table
            rot=[1, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path="path/to/cup.usd",  # see Cup Asset section below
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Table surface
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.15, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Ground plane and lighting (same as lift env)
    plane: AssetBaseCfg = ...
    light: AssetBaseCfg = ...
```

### Cup Asset Options

You need a USD of a small cup. Options:

1. **Isaac Sim props library**: Check `{ISAAC_NUCLEUS_DIR}/Props/` for existing cup/mug assets
2. **Simple cylinder primitive**: For initial testing, use a simple cylinder shape
3. **Objaverse**: Download a cup mesh from Objaverse and convert to USD
4. **Custom mesh**: Create in Blender/CAD, export as OBJ/STL, import in Isaac Sim

For initial development, use a **simple cylinder** as the cup proxy:

```python
from isaaclab.sim.spawners.shapes import CylinderCfg

object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.2, 0, 0.04],
        rot=[1, 0, 0, 0],
    ),
    spawn=CylinderCfg(
        radius=0.03,       # 3cm radius -- small cup
        height=0.08,       # 8cm tall
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # 50g cup
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.2, 0.2),  # Red cup for visibility
        ),
    ),
)
```

## Action Space

### Joint Position Control (Recommended for Phase 1)

```python
@configclass
class ActionsCfg:
    # Arm joints: direct position targets
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Roll", "Wrist_Pitch"],
        scale=0.5,           # scale down actions for smooth motion
        use_default_offset=True,
    )

    # Gripper: binary open/close
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["Jaw"],
        open_command_expr={"Jaw": 0.04},   # gripper open position
        close_command_expr={"Jaw": 0.0},   # gripper closed position
    )
```

### Differential IK Control (For Phase 2 / Vision)

For visuomotor policies, task-space (Cartesian) control via differential IK is often better:

```python
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

arm_action = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Roll", "Wrist_Pitch"],
    body_name="gripper_tip",  # end-effector body
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    ),
    scale=0.5,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
        pos=[0.0, 0.0, 0.0],  # offset from body to actual grasp point
    ),
)
```

## Observation Space

### Phase 1: State-Based Observations

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Robot state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)      # relative joint positions
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)      # joint velocities

        # Object state (privileged -- from simulator)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame  # cup position relative to robot
        )

        # Goal position (where to lift the cup)
        target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )

        # Previous actions (helps with temporal reasoning)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True  # add observation noise
            self.concatenate_terms = True
```

### Phase 2: Vision-Based Observations (see 06_VISION_POLICY.md)

Adds camera images to the observation space alongside state information.

## Reward Function Design

The reward function is critical for learning the cup-picking behavior. Based on the existing `lift` task rewards, here's a staged reward design:

### Reward Components

```python
@configclass
class RewardsCfg:
    # Stage 1: Reach the cup
    # Tanh-kernel reward -- decreases smoothly as EE gets closer to cup
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=1.0,
    )

    # Stage 2: Lift the cup
    # Binary reward when cup is above threshold height
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.06},  # 6cm above table
        weight=15.0,                       # high weight to prioritize lifting
    )

    # Stage 3: Track the goal position (lift to target height)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.06,
            "command_name": "object_pose",
        },
        weight=16.0,
    )

    # Stage 3b: Fine-grained goal tracking
    object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.06,
            "command_name": "object_pose",
        },
        weight=5.0,
    )

    # Penalties: smooth motion
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

### How the Rewards Guide Learning

1. **reaching_object** (weight 1.0): Gets the gripper close to the cup
2. **lifting_object** (weight 15.0): Strong signal when cup leaves the table
3. **object_goal_tracking** (weight 16.0): Moves cup toward goal position once lifted
4. **action_rate + joint_vel** (weight -1e-4, curriculum to -1e-1): Smooth motion

The curriculum (`CurriculumCfg`) gradually increases the action/velocity penalties over 10,000 steps, starting with exploration-friendly low penalties and moving to smooth-motion-encouraging higher penalties.

### Cup-Specific Reward Considerations

Unlike a cube, a cup has:
- **Top opening**: The gripper should approach from the side, not the top
- **Thin walls**: Grip force matters -- too much force could tip it over
- **Height**: Taller than a cube, so approach angle matters

Consider adding a custom reward for **grasp stability**:

```python
def cup_grasp_stable(env, object_cfg, ee_frame_cfg):
    """Reward for maintaining stable contact with the cup while lifted."""
    object = env.scene[object_cfg.name]
    # Check the cup orientation hasn't tipped too much
    cup_up = object.data.root_quat_w  # extract up vector
    # Reward for keeping the cup upright while lifted
    ...
```

## Termination Conditions

```python
@configclass
class TerminationsCfg:
    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Cup fell off the table
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
```

## Event Configuration (Randomization)

```python
@configclass
class EventCfg:
    # Reset scene to default on episode reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize cup position on table (within reachable area)
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),   # +/- 5cm from center
                "y": (-0.10, 0.10),   # +/- 10cm side to side
                "z": (0.0, 0.0),       # on the table surface
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
```

## Command Configuration

The command defines the **target position** for where the cup should be lifted to:

```python
@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_tip",  # SO-101 end-effector
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.25),    # in front of the arm
            pos_y=(-0.10, 0.10),   # side to side
            pos_z=(0.15, 0.30),    # above the table
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
```

## Full Environment Config

```python
@configclass
class SO101CupPickEnvCfg(ManagerBasedRLEnvCfg):
    scene: CupPickingSceneCfg = CupPickingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01  # 100Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
```

## File Structure for the Custom Task

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/
  cup_pick/
    __init__.py                # Gym environment registration
    cup_pick_env_cfg.py        # Base environment config
    mdp/
      __init__.py
      observations.py          # Custom observation functions
      rewards.py               # Custom reward functions (cup-specific)
      terminations.py          # Custom termination conditions
    config/
      so101/
        __init__.py            # Register Isaac-Pick-Cup-SO101-v0
        joint_pos_env_cfg.py   # SO-101 joint position control config
        agents/
          __init__.py
          rsl_rl_ppo_cfg.py    # PPO hyperparameters
```
