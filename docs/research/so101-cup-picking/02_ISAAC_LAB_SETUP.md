# Isaac Lab Setup: Installation & Configuration

## Prerequisites

| Requirement | Version |
|-------------|---------|
| **OS** | Ubuntu 22.04+ |
| **GPU** | NVIDIA GPU (RTX 3070+ recommended for training) |
| **NVIDIA Driver** | 535+ |
| **CUDA** | 12.x |
| **Python** | 3.11 |
| **Isaac Sim** | 5.1.0+ |
| **Isaac Lab** | 2.3.0+ |

## Installation Options

### Option A: pip Install (Recommended for Training)

```bash
# Create conda environment
conda create -n isaaclab python=3.11
conda activate isaaclab

# Install Isaac Sim
pip install 'isaacsim[all]==5.1.0' --extra-index-url https://pypi.nvidia.com

# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install Isaac Lab
./isaaclab.sh --install  # or isaaclab.bat on Windows
```

### Option B: From Source (This Repository)

Since you already have the Isaac Lab codebase at `/home/user/isaaclab`:

```bash
cd /home/user/isaaclab
./isaaclab.sh --install
```

### Install RL Libraries

```bash
# RSL-RL (recommended for manipulation)
./isaaclab.sh -p -m pip install rsl-rl-lib==3.0.1

# SKRL (alternative)
./isaaclab.sh -p -m pip install skrl

# RL Games (alternative)
./isaaclab.sh -p -m pip install rl-games
```

## Kit-Based vs Standalone Mode

You mentioned wanting to use the **Kit version** for visualization. Here's how the two modes compare:

### Standalone Mode (Default for Training)

```bash
# Train with visualization (slower, but you can see the training)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-OpenArm-v0

# Train headless (faster, no rendering)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-OpenArm-v0 --headless

# Train headless with video recording
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-OpenArm-v0 --headless --enable_cameras --video
```

### Kit-Based Mode (Interactive Visualization)

Isaac Lab supports running as a Kit extension, which gives you:
- Full Omniverse viewport with RTX rendering
- Hot-reloading of code changes
- Interactive inspection of the scene during training
- Better visual debugging

To use Kit mode, you run Isaac Sim as the Kit application and load Isaac Lab as an extension. The training scripts in Isaac Lab's standalone mode already support GUI visualization by default (without `--headless`). When you run training without `--headless`, it opens an Isaac Sim window where you can observe training in real-time.

**Recommendation for your workflow:**
1. **Development/debugging**: Run without `--headless` to see the visualization
2. **Long training runs**: Use `--headless` with `--video` for periodic video captures
3. **Final visualization**: Use the `play` script with the trained checkpoint

### Render Modes During Training

When running with the GUI visible, you can switch between render modes in the "Isaac Lab" window:
- **Full rendering**: RTX path-traced (beautiful but slow)
- **Viewport rendering**: Real-time rendering (good balance)
- **Headless with offscreen**: No GUI but records videos

## Verify Installation

```bash
# Test basic functionality
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# Test an existing manipulation environment
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-OpenArm-v0 --num_envs 16 --max_iterations 10
```

## Existing SO-101 Isaac Lab Project

The community project [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) provides a ready-made Isaac Lab external project for SO-101. It can be used as a starting point:

```bash
# Clone the community project
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101

# Install dependencies (uses uv package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Train reaching
uv run train --task SO-ARM100-Reach-v0 --headless

# Evaluate
uv run play --task SO-ARM100-Reach-Play-v0
```

Logs are saved to: `~/Documents/isaac_so_arm101/logs/rsl_rl/so_arm100_reach/`

## GPU Memory Considerations

| Num Envs | Approximate VRAM | Training Speed |
|----------|-------------------|----------------|
| 256 | ~4 GB | Moderate |
| 1024 | ~8 GB | Fast |
| 4096 | ~16 GB | Very fast (default) |
| 8192 | ~24 GB+ | Maximum throughput |

For the cup picking task with camera observations (Phase 2), VRAM usage increases significantly. Start with fewer environments:
- **State-based (Phase 1)**: 2048-4096 envs
- **Vision-based (Phase 2)**: 128-512 envs (cameras are expensive)
