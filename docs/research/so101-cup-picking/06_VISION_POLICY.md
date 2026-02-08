# Vision-Based Policy: Camera Observations & Visuomotor Training

## Overview

Phase 1 uses privileged state information (the simulator tells the policy where the cup is). For real-world deployment, the policy needs to **see** the cup using the camera mounted on the arm. This document covers how to add vision to the policy.

## Do You Need a Special Vision Model?

**Short answer: Not necessarily for Phase 1, but yes for Phase 2 deployment.**

### Approach Options

| Approach | Complexity | Sim-to-Real Quality | Training Cost |
|----------|------------|---------------------|---------------|
| **A: State-based only** | Low | Requires external perception | Fast (no cameras) |
| **B: End-to-end visuomotor** | High | Best potential transfer | Very slow (camera rendering) |
| **C: Teacher-student distillation** | Medium | Good transfer | Moderate |
| **D: Separate vision + state policy** | Medium | Good, modular | Moderate |

### Recommended Approach: C or D

**Approach C (Teacher-Student):**
1. Train a **teacher** policy using privileged state info (Phase 1)
2. Train a **student** policy that uses camera images to mimic the teacher
3. The student learns to extract relevant information from images

**Approach D (Separate Vision Module):**
1. Train the state-based RL policy (Phase 1)
2. Train a separate vision model (e.g., object detector or pose estimator) to detect the cup
3. Feed detected cup position to the state-based policy at inference time
4. This is simpler and more debuggable

## Camera Setup in Isaac Lab

Isaac Lab supports cameras as scene entities. Based on the visuomotor stack config:

### Wrist Camera (Mounted on SO-101 Gripper)

```python
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

# Camera mounted on the wrist (matching your real camera)
wrist_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
    update_period=0.0,        # update every sim step
    height=200,               # image height (pixels)
    width=200,                # image width (pixels)
    data_types=["rgb", "distance_to_image_plane"],  # RGB + depth
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 2.0),   # 1cm to 2m range
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.0),        # adjust to match your camera mount
        rot=(1.0, 0.0, 0.0, 0.0),    # adjust orientation
        convention="ros",
    ),
)
```

### External Table Camera (Optional, for training only)

```python
# Camera overlooking the workspace (like a third-person view)
table_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/table_cam",
    update_period=0.0,
    height=200,
    width=200,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 2.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.5, 0.0, 0.4),                              # above and in front
        rot=(0.35355, -0.61237, -0.61237, 0.35355),       # looking down at table
        convention="ros",
    ),
)
```

### Adding Cameras to Observations

```python
@configclass
class VisuomotorObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # State observations (proprioception)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)      # end-effector position
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)    # end-effector orientation
        gripper_pos = ObsTerm(func=mdp.gripper_pos)    # gripper state

        # Camera observations
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,   # keep raw pixels for CNN
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # images can't be concatenated with vectors
```

**Important**: When using image observations, set `concatenate_terms = False` because images have different dimensions than vector observations.

## Tiled Rendering for Efficient Camera Training

Isaac Lab's `TiledCamera` renders all camera views in a single GPU pass, making it much more efficient than rendering each camera separately:

```python
from isaaclab.sensors import TiledCameraCfg

wrist_cam = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
    update_period=0.0,
    height=84,           # smaller images for faster training
    width=84,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 2.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.05, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="ros",
    ),
)
```

Supported data types for TiledCamera:
- `rgb` - 3-channel color image
- `rgba` - 4-channel with alpha
- `distance_to_camera` - depth (distance to optical center)
- `distance_to_image_plane` - depth (distance to image plane)
- `depth` - alias for distance_to_image_plane
- `normals` - surface normal vectors
- `semantic_segmentation` - semantic labels
- `instance_segmentation_fast` - instance labels

## Policy Network Architecture for Vision

### CNN + MLP Architecture

For visuomotor policies, the network needs a CNN backbone to process images:

```
Camera Image (84x84x3)
    |
    v
CNN Backbone (ResNet-18 or custom)
    |
    v
Flatten -> 512-dim feature vector
    |
    v
Concatenate with state observations (joint pos, vel, etc.)
    |
    v
MLP (256, 128, 64) -> Action
```

### With RSL-RL

RSL-RL supports custom policy architectures. You would need to define a custom actor-critic that includes a CNN:

```python
# In your agent config
policy = RslRlPpoActorCriticCfg(
    class_name="ActorCriticCNN",  # custom class
    init_noise_std=1.0,
    cnn_channels=[32, 64, 64],
    cnn_kernel_sizes=[8, 4, 3],
    cnn_strides=[4, 2, 1],
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
)
```

### With SKRL

SKRL has built-in support for mixed observation spaces (images + vectors):

```python
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

class VisuomotorPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # CNN for image observations
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # MLP for combined features
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_size + state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )
```

## Teacher-Student Distillation (Recommended)

This is the most practical approach for sim-to-real:

### Step 1: Train Teacher (State-Based)

```bash
# Train the teacher with full state information (Phase 1 policy)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Pick-Cup-SO101-v0 \
    --num_envs 4096 --headless
```

### Step 2: Distill to Student (Vision-Based)

RSL-RL supports distillation via `DistillationRunner`:

```python
@configclass
class SO101CupPickDistillationCfg(RslRlDistillationRunnerCfg):
    class_name = "DistillationRunner"
    experiment_name = "so101_cup_pick_vision"

    # Teacher configuration
    teacher = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
    )

    # Student configuration (with CNN)
    student = RslRlPpoActorCriticCfg(
        class_name="ActorCriticCNN",
        cnn_channels=[32, 64, 64],
        actor_hidden_dims=[256, 128, 64],
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        teacher_checkpoint="path/to/teacher/model.pt",
        distillation_loss_coef=1.0,
    )
```

### Step 3: Fine-tune with RL (Optional)

After distillation, optionally fine-tune the student with RL rewards to improve performance beyond the teacher.

## Domain Randomization for Vision

Critical for camera-based sim-to-real transfer:

```python
@configclass
class VisionEventCfg(EventCfg):
    # Randomize lighting
    randomize_light = EventTerm(
        func=franka_stack_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (1500.0, 10000.0),
            "color_variation": 0.4,
            "textures": [
                # List of HDR sky textures
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                # ... more textures
            ],
        },
    )

    # Randomize table surface appearance
    randomize_table_visual = EventTerm(
        func=franka_stack_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "textures": [
                # Various wood/metal/stone textures
            ],
        },
    )

    # Randomize robot arm appearance
    randomize_robot_visual = EventTerm(
        func=franka_stack_events.randomize_visual_texture_material,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

Additional visual randomization to consider:
- Camera intrinsics (focal length, distortion)
- Camera extrinsics (small pose perturbations)
- Cup color and texture
- Background distractors
- Shadows and reflections

## Rendering Settings for Training

```python
# Rendering quality during training
self.sim.render.antialiasing_mode = "DLAA"
self.num_rerenders_on_reset = 3  # re-render 3 frames on reset for stable images

# Image observation list
self.image_obs_list = ["wrist_cam"]
```

## VRAM Considerations

Vision-based training is significantly more expensive:

| Configuration | VRAM Estimate |
|--------------|---------------|
| 128 envs, 84x84 RGB, 1 camera | ~8 GB |
| 256 envs, 84x84 RGB, 1 camera | ~12 GB |
| 128 envs, 200x200 RGB, 2 cameras | ~16 GB |
| 512 envs, 84x84 RGB, 1 camera | ~20 GB |

Start with **128 environments** and **84x84 images** for the wrist camera only.
