# Complete Step-by-Step Runbook: SO-101 Cup Picking

## Prerequisites Checklist

- [ ] Ubuntu 22.04+ with NVIDIA GPU (RTX 3070 or better)
- [ ] NVIDIA Driver 535+, CUDA 12.x installed
- [ ] Isaac Lab installed and working (verify with `./isaaclab.sh -p -c "print('OK')"`)
- [ ] RSL-RL installed (`pip install rsl-rl-lib==3.0.1`)
- [ ] SO-101 arm with camera (for eventual deployment)

---

## PHASE 1: State-Based Policy (Sim Only)

### Step 1: Get the SO-101 Model into Isaac Lab

**Option A: Quick Start with Community Project (Fastest)**

```bash
# Clone the existing SO-101 Isaac Lab project
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101

# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Verify it works with a reaching task
uv run train --task SO-ARM100-Reach-v0 --num_envs 16 --max_iterations 10
```

This gets you a working SO-101 in Isaac Lab immediately.

**Option B: Convert URDF (More Control)**

```bash
# 1. Download the SO-101 URDF
git clone https://github.com/TheRobotStudio/SO-ARM100.git

# 2. Examine the URDF
cat SO-ARM100/Simulation/SO101/so101_new_calib.urdf

# 3. Convert to USD
cd /path/to/isaaclab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    /path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
    /path/to/output/so101/so101.usd \
    --fix-base \
    --joint-stiffness 80.0 \
    --joint-damping 4.0 \
    --joint-target-type position

# 4. Verify in Isaac Sim (opens GUI)
./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py
# Manually load the generated USD and check joints work
```

**Option C: Check Isaac Sim Asset Library**

The SO-101 may already be available as a pre-built USD asset in Isaac Sim under:
```
{ISAAC_NUCLEUS_DIR}/Robots/RobotStudio/
```

### Step 2: Create the Cup-Picking Environment

Create a new task directory structure. The simplest approach is to adapt the
existing `lift` task. Use the OpenArm lift config as a template:

```
# Key source files to reference/copy:
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/
  lift_env_cfg.py                              # Base env config
  config/openarm/lift_openarm_env_cfg.py       # OpenArm-specific overrides
  config/openarm/joint_pos_env_cfg.py          # Joint position control
  config/openarm/agents/rsl_rl_ppo_cfg.py      # PPO hyperparameters
  mdp/rewards.py                               # Reward functions
```

**What to change from the OpenArm lift config:**

1. **Robot**: Replace `OPENARM_UNI_CFG` with your SO-101 `ArticulationCfg`
2. **Joint names**: Replace `openarm_joint.*` with SO-101 joint names
3. **End-effector frame**: Replace `openarm_ee_tcp` with SO-101 gripper tip
4. **Object**: Replace the DexCube with a cylinder (cup proxy)
5. **Positions**: Adjust object spawn position, command ranges for SO-101 workspace
6. **Gripper**: Adjust open/close positions for SO-101 jaw

Key environment parameters to adjust for the smaller SO-101:

```python
# Command ranges (target lift position) -- SO-101 has ~30cm reach
commands.object_pose.ranges.pos_x = (0.12, 0.22)   # within reach
commands.object_pose.ranges.pos_y = (-0.10, 0.10)
commands.object_pose.ranges.pos_z = (0.10, 0.25)   # above table

# Object spawn position -- on table, reachable
scene.object.init_state.pos = [0.18, 0, 0.04]  # in front of arm

# Object randomization range -- keep within workspace
events.reset_object_position.params.pose_range = {
    "x": (-0.04, 0.04),
    "y": (-0.06, 0.06),
    "z": (0.0, 0.0),
}
```

### Step 3: Register the Environment

In `__init__.py` of your task config directory:

```python
import gymnasium as gym

gym.register(
    id="Isaac-Pick-Cup-SO101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "path.to.your.env_cfg:SO101CupPickEnvCfg",
        "rsl_rl_cfg_entry_point": "path.to.your.agents.rsl_rl_ppo_cfg:SO101CupPickPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pick-Cup-SO101-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "path.to.your.env_cfg:SO101CupPickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "path.to.your.agents.rsl_rl_ppo_cfg:SO101CupPickPPORunnerCfg",
    },
    disable_env_checker=True,
)
```

### Step 4: Test the Environment (No Training)

```bash
# Run with a random agent to verify scene setup
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4

# Or using the community project:
uv run zero_agent --task SO-ARM100-Reach-v0
```

Verify:
- [ ] SO-101 arm spawns correctly on the table
- [ ] Cup/cylinder spawns in front of the arm
- [ ] Joints move when random actions are applied
- [ ] Gripper opens and closes
- [ ] Cup has physics (falls if pushed)
- [ ] No strange collision issues

### Step 5: Train the State-Based Policy

```bash
# Start training (with GUI to watch progress)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4096

# Or headless for max speed
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 3000
```

### Step 6: Monitor Training

```bash
# In another terminal
tensorboard --logdir logs/rsl_rl/so101_cup_pick/
```

Watch these metrics:
- `Train/mean_reward` -> should increase over time
- `Train/mean_episode_length` -> should increase (longer episodes = cup held longer)
- `Policy/mean_noise_std` -> should decrease (policy becomes more deterministic)

### Step 7: Evaluate the Trained Policy

```bash
# Play the trained policy (with visualization)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Pick-Cup-SO101-Play-v0 \
    --num_envs 50 \
    --load_run <timestamp_run_name>
```

**Success criteria for Phase 1:**
- [ ] Arm consistently reaches toward the cup
- [ ] Gripper closes around the cup
- [ ] Cup is lifted off the table
- [ ] Cup is held above the minimum height threshold

### Step 8: Iterate on Rewards (If Needed)

If the policy doesn't learn to grasp, try:

1. **Increase reaching reward weight** (from 1.0 to 2.0)
2. **Add a grasp bonus** reward when gripper contacts the cup
3. **Reduce action penalty** to allow more exploration
4. **Increase episode length** (from 5s to 8s) to give more time
5. **Reduce object randomization** range initially

---

## PHASE 2: Vision-Based Policy

### Step 9: Add Camera to the Environment

Modify the scene config to add a wrist camera:

```python
from isaaclab.sensors import CameraCfg

# Add to scene config
scene.wrist_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
    update_period=0.0,
    height=84,
    width=84,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 2.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="ros",
    ),
)
```

### Step 10: Choose Vision Strategy

**Option A: Teacher-Student Distillation (Recommended)**

1. Use the Phase 1 teacher policy
2. Train a student that uses camera images
3. Student learns to map images -> same actions as teacher

```bash
# Train distillation student
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-Vision-v0 \
    --num_envs 128 \
    --headless \
    --enable_cameras
```

**Option B: Separate Perception + State Policy**

1. Keep the Phase 1 state-based policy
2. Train a separate object detection / pose estimation model
3. At inference: camera -> detector -> cup position -> state policy -> action

This is simpler and more debuggable. You can use:
- A small CNN trained in sim to predict cup position from the image
- A pre-trained detection model (YOLO, etc.) fine-tuned on cup images
- Classical CV (color thresholding if the cup is a distinct color)

**Option C: End-to-End Visuomotor RL**

Train a policy that directly takes images and outputs actions. This requires:
- Fewer environments (128-256 due to camera cost)
- Longer training (more iterations needed)
- Heavy domain randomization

### Step 11: Add Domain Randomization for Sim-to-Real

Add to the event config:

```python
# Visual randomization
randomize_light = EventTerm(...)
randomize_table_texture = EventTerm(...)

# Physics randomization
randomize_object_mass = EventTerm(...)
randomize_actuator_gains = EventTerm(...)
randomize_joint_friction = EventTerm(...)
```

### Step 12: Train the Vision Policy

```bash
# With camera rendering enabled
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-Vision-v0 \
    --num_envs 128 \
    --headless \
    --enable_cameras \
    --max_iterations 5000

# Record videos periodically
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-Vision-v0 \
    --num_envs 128 \
    --headless \
    --enable_cameras \
    --video \
    --video_interval 1000
```

---

## PHASE 3: Real-Robot Deployment

### Step 13: Export the Policy

```bash
# Play and auto-export to ONNX
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Pick-Cup-SO101-Play-v0 \
    --num_envs 1 \
    --load_run <best_run>
```

Exported files at: `logs/rsl_rl/so101_cup_pick/<run>/exported/`
- `policy.pt` - PyTorch model
- `policy.onnx` - ONNX model for deployment

### Step 14: Calibrate the Real Robot

1. **Joint zero calibration**: Match simulation joint zeros to real servo positions
2. **Joint direction**: Verify all joints rotate in the correct direction
3. **Camera calibration**: Match simulation camera intrinsics/extrinsics to real camera
4. **Workspace verification**: Ensure the table, cup placement, and arm positioning match simulation

### Step 15: Deploy on Real Hardware

```python
import onnxruntime as ort
import numpy as np
from feetech_sdk import FeetechServoController  # or your servo SDK

# Load policy
session = ort.InferenceSession("policy.onnx")

# Initialize servo communication
controller = FeetechServoController(port="/dev/ttyUSB0")

# Control loop at 50Hz
import time
CONTROL_DT = 0.02  # 50Hz

while True:
    t_start = time.time()

    # Read current joint positions from servos
    joint_pos = controller.read_positions()

    # Build observation (matching training observation format)
    obs = np.concatenate([
        joint_pos - default_joint_pos,  # relative joint pos
        joint_vel,                       # joint velocities
        cup_position,                    # from vision or state
        target_position,                 # where to lift
        last_action,                     # previous action
    ]).astype(np.float32).reshape(1, -1)

    # Run inference
    actions = session.run(None, {"observations": obs})[0][0]

    # Apply actions to servos (with safety limits)
    target_positions = default_joint_pos + actions * action_scale
    target_positions = np.clip(target_positions, joint_min, joint_max)
    controller.write_positions(target_positions)

    # Maintain control rate
    elapsed = time.time() - t_start
    if elapsed < CONTROL_DT:
        time.sleep(CONTROL_DT - elapsed)
```

### Step 16: Iterate and Improve

1. **Record failure cases** on the real robot
2. **Adjust domain randomization** to cover real-world conditions
3. **Fine-tune** the policy if needed
4. **Consider LeRobot integration** for a more polished deployment pipeline

---

## Quick Reference: Key Commands

```bash
# Convert URDF
./isaaclab.sh -p scripts/tools/convert_urdf.py INPUT.urdf OUTPUT.usd --fix-base

# Train (state-based)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 --num_envs 4096 --headless

# Train (vision-based)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-Vision-v0 --num_envs 128 --headless --enable_cameras

# Evaluate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Pick-Cup-SO101-Play-v0 --load_run <run>

# Monitor
tensorboard --logdir logs/rsl_rl/so101_cup_pick/
```

## Estimated Timeline (Not Wall-Clock Predictions)

| Phase | Steps | Description |
|-------|-------|-------------|
| **Setup** | 1-4 | Get SO-101 in Isaac Lab, create environment, verify scene |
| **Phase 1 Training** | 5-8 | Train state-based policy, iterate on rewards |
| **Phase 2 Vision** | 9-12 | Add cameras, domain randomization, train vision policy |
| **Phase 3 Deploy** | 13-16 | Export, calibrate, deploy, iterate |

## Resources and References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [SO-ARM100 GitHub (URDF/Simulation files)](https://github.com/TheRobotStudio/SO-ARM100)
- [isaac_so_arm101 (Community Isaac Lab project)](https://github.com/MuammerBay/isaac_so_arm101)
- [LeRobot SO-101 Docs](https://huggingface.co/docs/lerobot/so101)
- [Seeed Studio SO-101 Training Wiki](https://wiki.seeedstudio.com/training_soarm101_policy_with_isaacLab/)
- [LycheeAI Tutorial Series](https://lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series)
- [Isaac Lab Sim-to-Real Deployment](https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/index.html)
- [NVIDIA Blog: Sim-to-Real for Assembly](https://developer.nvidia.com/blog/bridging-the-sim-to-real-gap-for-industrial-robotic-assembly-applications-using-nvidia-isaac-lab/)
- [Isaac Lab Reference Architecture](https://isaac-sim.github.io/IsaacLab/main/source/refs/reference_architecture/index.html)
- [SO-100 Cube Lifting with SKRL (Medium)](https://medium.com/@kabilankb2003/training-so-100-robot-for-cube-lifting-in-isaac-lab-from-simulation-to-intelligent-control-with-9e81f94c6d6e)
