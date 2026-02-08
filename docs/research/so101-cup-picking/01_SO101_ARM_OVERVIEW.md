# SO-101 Arm: Hardware Specifications & Resources

## What is the SO-101?

The **SO-101** (Standard Open Arm 101) is a 6-DOF open-source robotic arm designed by [TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100) in collaboration with Hugging Face. It is the next-generation version of the SO-100, designed for AI robotics research and affordable manipulation experiments.

### Key Specifications

| Property | Value |
|----------|-------|
| **Degrees of Freedom** | 6 (5 arm joints + 1 gripper) |
| **Servo Motors** | FEETECH STS3215 (all joints) |
| **Follower Arm Torque** | 30 kg.cm (12V version) or 16.5 kg.cm (7.4V) |
| **Leader Arm Voltage** | 7.4V (always) |
| **Gripper Type** | Parallel jaw (moving jaw) |
| **Communication** | Serial (FEETECH protocol) |
| **Controller Board** | FEETECH SCS/STS servo bus |
| **Approximate Cost** | ~$130 (3D-printed) to ~$240 (kit) |
| **Weight** | ~500g (arm only) |
| **Reach** | ~30cm |

### Joint Configuration (from URDF)

The SO-101 has the following joints:

| Joint | Name | Type | Description |
|-------|------|------|-------------|
| 1 | Base Rotation | Revolute | Rotates the entire arm |
| 2 | Shoulder Pitch | Revolute | Lifts/lowers the upper arm |
| 3 | Elbow Pitch | Revolute | Bends the forearm |
| 4 | Wrist Roll | Revolute | Rolls the wrist |
| 5 | Wrist Pitch | Revolute | Pitches the wrist |
| 6 | Gripper | Revolute | Opens/closes the jaw |

### Motor Gear Ratios (SO-101 Leader Arm)

- 3x STS3215 with 1/147 gear ratio
- 2x STS3215 with 1/191 gear ratio
- 1x STS3215 with 1/345 gear ratio

### Improvements Over SO-100

- Improved wiring to prevent disconnection issues (previously seen at joint 3)
- Motors with optimized gear ratios for leader arm
- Leader arm can now follow the follower arm in real-time (useful for RL with human-in-the-loop corrections)
- Easier assembly (no gear removal step)

## URDF Sources

### 1. Official Repository (Recommended)

```
https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
```

This directory contains:
- `so101_new_calib.urdf` - URDF with new calibration (joint zero = middle of range)
- `so101_old_calib.urdf` - URDF with old calibration (joint zero = fully extended)
- MuJoCo XML files (`so101_new_calib.xml`, `so101_old_calib.xml`)
- Mesh files (STL/OBJ) for visual and collision geometry

**Important**: The generated URDFs use relative mesh paths (not `package://`), making them easier to use in Isaac Sim.

### 2. Hugging Face Hosted URDF

```
https://huggingface.co/haixuantao/dora-bambot/blob/main/URDF/so101.urdf
```

### 3. Community-Fixed URDF

The official URDF has known issues ([GitHub Issue #54](https://github.com/TheRobotStudio/SO-ARM100/issues/54)):
- Missing joint limits
- Some inverted joint axes
- Lack of decomposed collision meshes

The [brukg/SO-100-arm](https://github.com/brukg/SO-100-arm) repository provides a cleaned-up version with:
- Proper joint limits in `joint_limits.yaml`
- Correct joint axes
- MoveIt2 integration

### 4. Isaac Sim Official Assets

The SO-100/SO-101 models from RobotStudio are **officially included in Isaac Sim's robot assets** at:
```
{ISAAC_NUCLEUS_DIR}/Robots/RobotStudio/
```

This means you may not even need to convert the URDF yourself -- check if the USD is already available in the Isaac Sim asset library.

## URDF Calibration Note

The URDF zero positions differ from the LeRobot zero positions convention. When calibrating for sim-to-real:

- **New Calibration**: Each joint's virtual zero = middle of joint range (recommended for RL)
- **Old Calibration**: Each joint's virtual zero = fully extended horizontal position

For RL training, the **new calibration** is recommended as it provides more symmetric joint limits.

## Comparison with OpenArm (Already in Isaac Lab)

Isaac Lab already includes the **OpenArm** robot, which is similar but different:

| Feature | SO-101 | OpenArm |
|---------|--------|---------|
| DOF (arm) | 5 | 7 |
| Gripper | Parallel jaw (1 DOF) | Parallel jaw (mimic) |
| Servo Type | FEETECH STS3215 | Damiao DM-J series |
| Torque | ~30 kg.cm | 7-40 Nm (joint-dependent) |
| Size | Desktop (~30cm reach) | Larger |
| USD Available | Yes (Isaac Sim assets) | Yes (Isaac Nucleus) |

The OpenArm lift task (`Isaac-Lift-Cube-OpenArm-v0`) in Isaac Lab is the best starting template -- we will adapt its environment configuration to work with the SO-101.

## Software Ecosystem

| Tool | Purpose |
|------|---------|
| [LeRobot](https://github.com/huggingface/lerobot) | Hugging Face framework for robot learning (imitation learning, RL) |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | GPU-accelerated RL training in simulation |
| [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) | Community Isaac Lab project for SO-101 reaching |
| [ROS2 + MoveIt2](https://github.com/brukg/SO-100-arm) | Traditional planning and control |
