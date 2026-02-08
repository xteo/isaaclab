# URDF to USD Conversion for the SO-101

## Overview

Isaac Lab requires robot models in **USD (Universal Scene Description)** format. The SO-101 arm is available as a URDF, which needs to be converted. There are multiple paths to get the SO-101 into Isaac Lab.

## Option 1: Use Existing Isaac Sim Asset (Easiest)

The SO-100/SO-101 may already be available as a USD in Isaac Sim's asset library:

```
{ISAAC_NUCLEUS_DIR}/Robots/RobotStudio/
```

Check by browsing the Isaac Sim Content Browser or running:

```bash
./isaaclab.sh -p -c "from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR; print(ISAAC_NUCLEUS_DIR)"
```

Then browse that directory in Isaac Sim's content browser for RobotStudio assets.

## Option 2: Use the Community isaac_so_arm101 Project

The [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) project already has the SO-101 working in Isaac Lab. It includes the converted USD asset and articulation configuration. This is the fastest path to getting started.

## Option 3: Convert URDF Yourself

### Step 1: Download the URDF

```bash
# Clone the official SO-ARM100 repository
git clone https://github.com/TheRobotStudio/SO-ARM100.git
cd SO-ARM100/Simulation/SO101

# The URDF files are here:
# - so101_new_calib.urdf (recommended -- joint zero at mid-range)
# - so101_old_calib.urdf (joint zero at fully extended)
```

### Step 2: Verify URDF Structure

Before converting, inspect the URDF to understand the joint/link structure:

```bash
# Check joint names and limits
grep -E '<joint|<limit' SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

Expected joints:
- `Rotation` (base rotation)
- `Pitch` (shoulder)
- `Elbow` (elbow flex)
- `Wrist_Roll` (wrist roll)
- `Wrist_Pitch` (wrist pitch)
- `Jaw` (gripper open/close)

### Step 3: Convert Using Isaac Lab's URDF Converter

Isaac Lab provides a built-in URDF-to-USD converter:

```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    /path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
    /path/to/output/so101.usd \
    --fix-base \
    --joint-stiffness 80.0 \
    --joint-damping 4.0 \
    --joint-target-type position
```

**Flags explained:**
- `--fix-base`: Fixes the base link to the world (the arm is mounted on a table)
- `--joint-stiffness 80.0`: PD controller stiffness (tune for your servos)
- `--joint-damping 4.0`: PD controller damping
- `--joint-target-type position`: Position control (matches STS3215 servos)

### Step 4: Convert Using Isaac Sim GUI (Alternative)

1. Open Isaac Sim
2. Go to **Isaac Utils -> Workflows -> URDF Importer**
3. Browse to the SO-101 URDF file
4. Set parameters:
   - Fix base: checked
   - Merge fixed joints: optional
   - Joint stiffness: 80.0
   - Joint damping: 4.0
5. Click Import
6. Save the resulting stage as a USD file

### Step 5: Verify the USD

After conversion, verify the model works correctly:

```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py \
    --robot /path/to/output/so101.usd
```

Or open it in Isaac Sim and:
1. Check that all joints move correctly
2. Verify collision meshes are reasonable
3. Test that the gripper opens and closes
4. Check that the arm doesn't self-collide unexpectedly

## Known Issues and Fixes

### Issue 1: Missing Joint Limits

The official URDF may have missing or incorrect joint limits. If the converted USD has joints that spin freely, you need to add joint limits manually:

```python
# In your ArticulationCfg, you can set soft limits:
soft_joint_pos_limit_factor=0.95  # Use 95% of the joint range
```

Or edit the URDF before conversion to add proper `<limit>` elements.

### Issue 2: Inverted Joint Axes

Some joint axes may be inverted compared to the real hardware. Test by:
1. Commanding positive angles in simulation
2. Comparing with the real robot's behavior
3. If inverted, flip the axis in the URDF (`<axis xyz="0 0 1"/>` to `<axis xyz="0 0 -1"/>`)

### Issue 3: Collision Mesh Quality

The default collision meshes may be too detailed (slow) or too simple (inaccurate). For RL training:
- Use convex decomposed collision meshes
- In Isaac Sim, right-click the mesh -> Properties -> Collision -> set approximation to "Convex Decomposition"

### Issue 4: Base Collision

The base collision mesh can cause problems where the arm collides with its own base. The Simulation directory README notes that base collision meshes were removed for this reason. Verify this in your converted USD.

## ArticulationCfg for SO-101

Once you have the USD, create an `ArticulationCfg` similar to the OpenArm config. Reference the existing OpenArm config at:
`source/isaaclab_assets/isaaclab_assets/robots/openarm.py`

Example SO-101 configuration:

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/so101.usd",  # or ISAAC_NUCLEUS path
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Adjust these based on the URDF joint names
            "Rotation": 0.0,
            "Pitch": 0.0,
            "Elbow": 0.0,
            "Wrist_Roll": 0.0,
            "Wrist_Pitch": 0.0,
            "Jaw": 0.04,  # gripper open
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Rotation", "Pitch", "Elbow", "Wrist_Roll", "Wrist_Pitch"],
            # STS3215 servo characteristics
            velocity_limit_sim=2.0,      # rad/s (approximate)
            effort_limit_sim=3.0,        # Nm (approximate for STS3215)
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            velocity_limit_sim=0.5,
            effort_limit_sim=2.0,
            stiffness=500.0,
            damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
```

**Note**: The exact joint names and limits depend on the specific URDF version you use. Verify these against the actual URDF file.
