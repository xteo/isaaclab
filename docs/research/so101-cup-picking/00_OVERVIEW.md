# SO-101 Cup Picking: RL Training Research & Implementation Plan

## Project Goal

Train a reinforcement learning policy for the **SO-101 robot arm** (6-DOF, with gripper) to **detect, reach, grasp, and lift a small cup** placed in front of it on a table. The policy should eventually transfer to the real robot using the on-arm camera for vision-based object detection.

## Document Index

| Document | Description |
|----------|-------------|
| [01_SO101_ARM_OVERVIEW.md](./01_SO101_ARM_OVERVIEW.md) | SO-101 arm specifications, URDF sources, and hardware details |
| [02_ISAAC_LAB_SETUP.md](./02_ISAAC_LAB_SETUP.md) | Isaac Lab installation, Kit vs Standalone, and environment setup |
| [03_URDF_TO_USD_CONVERSION.md](./03_URDF_TO_USD_CONVERSION.md) | Converting the SO-101 URDF to USD for Isaac Sim |
| [04_ENVIRONMENT_DESIGN.md](./04_ENVIRONMENT_DESIGN.md) | Custom cup-picking environment: scene, observations, actions, rewards |
| [05_TRAINING_PIPELINE.md](./05_TRAINING_PIPELINE.md) | RL training with RSL-RL / SKRL using PPO, hyperparameters, commands |
| [06_VISION_POLICY.md](./06_VISION_POLICY.md) | Adding camera observations, visuomotor policies, teacher-student |
| [07_SIM_TO_REAL.md](./07_SIM_TO_REAL.md) | Domain randomization, ONNX export, real-robot deployment |
| [08_STEP_BY_STEP_RUNBOOK.md](./08_STEP_BY_STEP_RUNBOOK.md) | Complete sequential runbook from zero to trained policy |

## Architecture Summary

```
                    +------------------+
                    |   SO-101 URDF    |
                    | (TheRobotStudio) |
                    +--------+---------+
                             |
                    convert_urdf.py
                             |
                    +--------v---------+
                    |   SO-101 USD     |
                    |  (Isaac Sim)     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+     +-----------v-----------+
    | Phase 1: State-    |     | Phase 2: Vision-      |
    | Based Policy       |     | Based Policy          |
    | (joint pos + obj   |     | (wrist cam + table    |
    |  pos as obs)       |     |  cam as obs)          |
    +---------+----------+     +-----------+-----------+
              |                             |
              +-------------+---------------+
                            |
                   +--------v--------+
                   |  ONNX Export    |
                   |  + Deployment   |
                   +--------+--------+
                            |
                   +--------v--------+
                   | Real SO-101 Arm |
                   | + Camera        |
                   +-----------------+
```

## Key Finding: Existing Reference Implementations

The research identified several key resources:

1. **Isaac Lab already has a `Lift` task** (`Isaac-Lift-Cube-OpenArm-v0`) for the OpenArm robot that lifts a cube -- this is the closest template to what we need
2. **The SO-101 URDF exists** in the [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) repository
3. **A community project** ([isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)) already implements SO-101 reaching in Isaac Lab
4. **Isaac Lab has visuomotor support** with `CameraCfg` for wrist and table cameras (see `stack_ik_rel_visuomotor_env_cfg.py`)
5. **Isaac Lab exports ONNX** models for real-robot deployment

## Recommended Approach: Two Phases

### Phase 1: State-Based RL (get the arm working)
- Use privileged state information (object position from simulator)
- Train with PPO using RSL-RL or SKRL
- Reward: reach object -> grasp -> lift above threshold
- This proves the task is solvable and gets a working baseline

### Phase 2: Vision-Based RL (make it deployable)
- Add wrist camera (matching your real camera)
- Use teacher-student distillation or direct visuomotor training
- Domain randomization for sim-to-real transfer
- Export to ONNX for deployment on Jetson/PC controlling the real arm
