# Sim-to-Real Transfer

## Overview

The ultimate goal is to deploy the trained policy on the real SO-101 arm with its camera. This document covers the complete pipeline from trained simulation policy to real-world deployment.

## The Sim-to-Real Gap

Key differences between simulation and reality that must be bridged:

| Gap Type | Simulation | Reality | Mitigation |
|----------|------------|---------|------------|
| **Physics** | Perfect rigid-body dynamics | Friction, backlash, flex | Domain randomization |
| **Actuators** | Ideal PD controllers | STS3215 servo response | Actuator modeling + DR |
| **Sensing** | Perfect state information | Noisy servos + camera | Observation noise |
| **Visual** | Rendered images | Real camera images | Visual DR + real data |
| **Objects** | Simulated cup | Real cup | Object randomization |
| **Timing** | Deterministic | Variable latency | Control rate matching |

## Domain Randomization Strategy

### 1. Physics Randomization

```python
@configclass
class SimToRealEventCfg:
    # Randomize object mass
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.03, 0.1),  # 30g to 100g cup
            "operation": "abs",
        },
    )

    # Randomize object friction
    randomize_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.1),
        },
    )

    # Randomize actuator gains (mimics servo variability)
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),  # +/- 20%
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # Randomize joint parameters
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "abs",
        },
    )

    # Add observation noise (mimics noisy servo feedback)
    # Handled via observation corruption in ObservationsCfg
```

### 2. Visual Randomization (For Camera Policies)

See [06_VISION_POLICY.md](./06_VISION_POLICY.md) for visual domain randomization including:
- Lighting randomization (intensity, color, HDR textures)
- Table surface texture randomization
- Cup color/texture randomization
- Robot arm appearance randomization
- Camera parameter randomization

### 3. Observation Noise

```python
@configclass
class PolicyCfg(ObsGroup):
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=mdp.UniformNoiseCfg(n_min=-0.01, n_max=0.01),  # +/- 0.01 rad
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=mdp.UniformNoiseCfg(n_min=-0.1, n_max=0.1),    # +/- 0.1 rad/s
    )

    def __post_init__(self):
        self.enable_corruption = True  # enables observation noise
```

## Policy Export

### ONNX Export (Recommended for Deployment)

After training, export the policy to ONNX format for deployment:

```bash
# Play and export
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Pick-Cup-SO101-Play-v0 \
    --num_envs 1 \
    --load_run <run_name>
```

The play script automatically exports:
- `policy.pt` - PyTorch checkpoint
- `policy.onnx` - ONNX model for deployment

These are saved to: `logs/rsl_rl/so101_cup_pick/<run>/exported/`

### Manual Export

```python
import torch

# Load the trained model
checkpoint = torch.load("logs/rsl_rl/so101_cup_pick/<run>/model_3000.pt")

# Create dummy input matching observation size
dummy_obs = torch.zeros(1, obs_dim)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_obs,
    "so101_cup_pick_policy.onnx",
    input_names=["observations"],
    output_names=["actions"],
    opset_version=11,
)
```

### JIT Export (Alternative)

```python
# Export as TorchScript JIT
scripted_model = torch.jit.script(model)
scripted_model.save("so101_cup_pick_policy.jit")
```

## Real-Robot Deployment Architecture

```
+------------------+      +------------------+      +-------------------+
|  Camera          | ---> | Vision Module    | ---> | Policy Network    |
|  (on arm)        |      | (optional CNN    |      | (ONNX/JIT model)  |
|                  |      |  or detector)    |      |                   |
+------------------+      +------------------+      +--------+----------+
                                                             |
                                                    Action (joint targets)
                                                             |
                                                    +--------v----------+
                                                    | State Estimator   |
                                                    | (servo feedback)  |
                                                    +--------+----------+
                                                             |
                                                    +--------v----------+
                                                    | Action Controller |
                                                    | (PD / position    |
                                                    |  control loop)    |
                                                    +--------+----------+
                                                             |
                                                    +--------v----------+
                                                    | STS3215 Servos    |
                                                    | (FEETECH bus)     |
                                                    +-------------------+
```

### Deployment Options

#### Option A: Python on Linux PC / Jetson

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("so101_cup_pick_policy.onnx")

# Main control loop at 50Hz (matching sim decimation)
while running:
    # 1. Read servo positions
    joint_positions = read_servo_positions()  # from FEETECH SDK

    # 2. Read camera (if vision policy)
    camera_image = capture_camera_frame()

    # 3. Build observation vector
    obs = build_observation(joint_positions, camera_image)

    # 4. Run policy inference
    actions = session.run(None, {"observations": obs})[0]

    # 5. Convert actions to servo commands
    target_positions = actions_to_servo_targets(actions)

    # 6. Send to servos
    send_servo_commands(target_positions)

    # 7. Wait for next control step
    time.sleep(0.02)  # 50Hz
```

#### Option B: LeRobot Integration

LeRobot provides a unified robot control interface that works with the SO-101:

```python
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

robot = ManipulatorRobot(
    robot_type="so101",
    # ... configuration
)

# Control loop
while running:
    state = robot.capture_observation()
    action = policy.select_action(state)
    robot.send_action(action)
```

#### Option C: ROS2 Integration

For ROS2-based deployment (useful if you want MoveIt2 safety constraints):

```bash
# Launch ROS2 control for SO-101
ros2 launch so100_arm so100_arm.launch.py

# Run inference node
ros2 run so101_inference policy_node \
    --model_path so101_cup_pick_policy.onnx
```

## Control Frequency Matching

The simulation runs at:
- Physics: 100Hz (dt=0.01)
- Policy: 50Hz (decimation=2)

The real robot should match:
- Servo read/write: 100Hz+ (STS3215 supports fast serial)
- Policy inference: 50Hz
- Camera capture: 30Hz (if using vision)

If using ONNX on a Jetson Orin, inference is typically 1-2ms, leaving plenty of time for the control loop.

## Calibration Checklist

Before deploying on the real robot:

- [ ] **Joint zero calibration**: Ensure URDF joint zeros match real servo zeros
- [ ] **Joint direction**: Verify positive rotation direction matches
- [ ] **Joint limits**: Confirm the policy respects real joint limits
- [ ] **Servo gains**: Tune PD gains to match simulation actuator model
- [ ] **Camera calibration**: Match intrinsics between sim and real camera
- [ ] **Camera mounting**: Match extrinsics (position, orientation) precisely
- [ ] **Control rate**: Verify real control loop runs at 50Hz consistently
- [ ] **Action scaling**: Check that policy actions map correctly to servo targets
- [ ] **Observation normalization**: Use the same normalization as training

## Safety Considerations

- Start with **slow motions** (reduce action scale to 0.1x) and gradually increase
- Implement **joint limit stops** in the control code (don't rely solely on the policy)
- Add **force/current limits** on the servos to prevent damage
- Test with the arm **unloaded** first (no cup, no table collision)
- Keep an **emergency stop** (power kill) accessible
- Monitor **servo temperatures** during extended operation
