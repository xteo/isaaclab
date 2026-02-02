---
name: imitation-learning
description: Guide for setting up imitation learning with robomimic or IsaacLab-Mimic. Use when user wants to learn from demonstrations, collect demonstration data, or train behavior cloning policies.
---

# Imitation Learning in Isaac Lab

## Overview

Isaac Lab supports imitation learning through two main frameworks:

1. **robomimic** - Standard IL library for behavior cloning
2. **IsaacLab-Mimic** - NVIDIA's IL module with motion generation

## Method 1: robomimic Integration

### Installation

```bash
# Install robomimic with Isaac Lab
./isaaclab.sh -i robomimic
```

### Step 1: Collect Demonstrations

Using teleoperation:
```bash
# Keyboard teleoperation
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Stack-Cube-Franka-v0 \
    --teleop_device keyboard \
    --dataset_file demos.hdf5 \
    --num_demos 50
```

Using motion planning:
```bash
# Scripted demonstration collection
./isaaclab.sh -p scripts/imitation_learning/robomimic/collect_demos.py \
    --task Isaac-Stack-Cube-Franka-v0 \
    --num_demos 1000 \
    --dataset_file demos.hdf5
```

### Step 2: Train Behavior Cloning

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --algo bc \
    --dataset ./demos.hdf5 \
    --num_epochs 2000
```

**Available algorithms:**
- `bc` - Vanilla Behavior Cloning
- `bc_rnn` - BC with recurrent networks
- `bcq` - Batch Constrained Q-learning (offline RL)
- `iql` - Implicit Q-Learning (offline RL)

### Step 3: Evaluate Policy

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --checkpoint logs/.../model.pt \
    --num_episodes 50
```

## Method 2: IsaacLab-Mimic

IsaacLab-Mimic provides motion generation and sim-to-sim transfer capabilities.

### Setup

```bash
# Install mimic module
./isaaclab.sh -e isaaclab_mimic
```

### Collect Demonstrations

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/collect.py \
    --task Isaac-Reach-Franka-v0 \
    --num_demos 100
```

### Train with Mimic

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/train.py \
    --task Isaac-Reach-Franka-v0 \
    --dataset collected_demos.hdf5
```

## Dataset Format (HDF5)

robomimic uses HDF5 format:

```
demos.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/           # Observations
│   │   │   ├── joint_pos  # (T, N_joints)
│   │   │   ├── joint_vel  # (T, N_joints)
│   │   │   └── ...
│   │   ├── actions        # (T, action_dim)
│   │   ├── rewards        # (T,)
│   │   ├── dones          # (T,)
│   │   └── states         # (T, state_dim) - optional
│   ├── demo_1/
│   └── ...
└── env_args/              # Environment configuration
```

### Creating Custom Datasets

```python
import h5py
import numpy as np

with h5py.File("my_demos.hdf5", "w") as f:
    data_grp = f.create_group("data")

    for i, demo in enumerate(demonstrations):
        demo_grp = data_grp.create_group(f"demo_{i}")

        # Store observations
        obs_grp = demo_grp.create_group("obs")
        obs_grp.create_dataset("joint_pos", data=demo["joint_pos"])
        obs_grp.create_dataset("joint_vel", data=demo["joint_vel"])

        # Store actions
        demo_grp.create_dataset("actions", data=demo["actions"])

        # Store rewards and dones
        demo_grp.create_dataset("rewards", data=demo["rewards"])
        demo_grp.create_dataset("dones", data=demo["dones"])
```

## Environment Configuration for IL

### IK-Based Actions (Recommended for IL)

```python
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg

@configclass
class ActionsCfg(ActionGroupCfg):
    # Delta end-effector control (easier to learn)
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,  # Delta commands
            ik_method="dls",
        ),
        scale=0.05,
    )
```

### Joint Position Actions (Alternative)

```python
from isaaclab.envs.mdp.actions import JointPositionActionCfg

@configclass
class ActionsCfg(ActionGroupCfg):
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.1,
        use_default_offset=True,
    )
```

## Observation Configuration for IL

### Key Observations for Manipulation

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Robot state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # End-effector state
        ee_pos = ObsTerm(func=mdp.body_position_in_root_frame,
                        params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")})
        ee_quat = ObsTerm(func=mdp.body_orientation_in_root_frame,
                         params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")})

        # Object state
        object_pos = ObsTerm(func=mdp.root_pos_w,
                            params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(func=mdp.root_quat_w,
                             params={"asset_cfg": SceneEntityCfg("object")})

        # Relative positions (important for generalization)
        ee_to_object = ObsTerm(func=custom_ee_to_object_pos)

    policy: PolicyCfg = PolicyCfg()
```

## Training Configuration

### robomimic Config

```python
# In training script or config file
config = {
    "algo_name": "bc",
    "experiment": {
        "name": "manipulation_bc",
        "validate": True,
        "save": {
            "enabled": True,
            "every_n_epochs": 50,
        },
    },
    "train": {
        "num_epochs": 2000,
        "batch_size": 100,
        "learning_rate": 1e-4,
        "seq_length": 10,  # For RNN models
    },
    "observation": {
        "modalities": {
            "obs": {
                "low_dim": ["joint_pos", "joint_vel", "ee_pos", "object_pos"],
            },
        },
        "encoder": {
            "low_dim": {
                "core_kwargs": {
                    "hidden_dim": 256,
                    "num_layers": 3,
                },
            },
        },
    },
    "algo": {
        "gmm": {  # For BC-GMM
            "enabled": False,
            "num_modes": 5,
        },
        "rnn": {  # For BC-RNN
            "enabled": False,
            "hidden_dim": 400,
            "num_layers": 2,
        },
    },
}
```

## Best Practices

### Demonstration Quality

1. **Diversity** - Cover different starting positions, object locations
2. **Quantity** - More demos generally help (100-1000 typical)
3. **Consistency** - Remove failed demonstrations
4. **Noise** - Small action noise can help generalization

### Observation Design

1. **Relative positions** - More generalizable than absolute
2. **Proprioception** - Always include joint state
3. **History** - RNN models can use temporal context
4. **Normalization** - Normalize observations to similar scales

### Action Design

1. **Delta actions** - Easier to learn than absolute
2. **IK actions** - More intuitive for manipulation
3. **Action scaling** - Keep actions in reasonable range
4. **Smooth demonstrations** - Avoid jerky motions

## Debugging IL

### Check Dataset

```python
import h5py

with h5py.File("demos.hdf5", "r") as f:
    # Check structure
    print("Demos:", list(f["data"].keys()))

    # Check first demo
    demo = f["data/demo_0"]
    print("Observations:", list(demo["obs"].keys()))
    print("Action shape:", demo["actions"].shape)

    # Check value ranges
    actions = demo["actions"][:]
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
```

### Verify Environment Compatibility

```python
# Ensure env matches dataset
import gymnasium as gym

env = gym.make("Isaac-MyTask-v0")
obs, _ = env.reset()

# Check obs matches dataset structure
print(f"Env obs shape: {obs.shape}")
print(f"Env action dim: {env.action_space.shape}")
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Policy doesn't move | Action scale mismatch | Check demo action range vs env |
| Erratic behavior | Observation mismatch | Verify obs ordering matches |
| Low success rate | Insufficient demos | Collect more diverse demos |
| Overfitting | Too few demos or too many epochs | Early stopping, more data |

## Advanced: Diffusion Policy

For state-of-the-art IL, consider Diffusion Policy:

```bash
# Install diffusion policy
pip install diffusers

# Training (requires robomimic setup)
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --algo diffusion_policy \
    --dataset ./demos.hdf5
```

## Workflow Summary

```
1. Design Environment
   └── IK-based actions, relevant observations

2. Collect Demonstrations
   ├── Teleoperation (human demos)
   └── Motion planning (automated)

3. Preprocess Data
   ├── Filter failed demos
   ├── Normalize observations
   └── Convert to HDF5

4. Train BC Policy
   ├── Start with vanilla BC
   └── Try BC-RNN or Diffusion if needed

5. Evaluate
   ├── Check success rate
   └── Analyze failure modes

6. Iterate
   ├── Collect more demos
   └── Adjust observations/actions
```
