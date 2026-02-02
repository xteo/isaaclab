---
name: debugging-simulation
description: Diagnose and fix common Isaac Lab issues. Use when user reports errors, unexpected behavior, training problems, physics issues, NaN values, or simulation instability.
---

# Debugging Isaac Lab Simulations

## Quick Diagnostic Tree

```
Problem?
├── Simulation crashes
│   ├── OOM → Reduce num_envs, check GPU memory
│   ├── NaN → Check reward values, physics params
│   └── CUDA error → Check device config, driver
├── Robot falls through ground
│   └── Check collision filters, physics material
├── Robot explodes/jitters
│   ├── Check actuator gains (too high)
│   └── Check timestep (too large)
├── Training not improving
│   ├── Check reward values (all zeros?)
│   ├── Check observation values (NaN? constant?)
│   └── Check action effect (robot moving?)
└── Training unstable
    ├── Check learning rate
    └── Check reward scale/variance
```

## Issue: Simulation Crashes

### Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce `num_envs` (start with 256-512)
2. Reduce observation/action dimensions
3. Use smaller network architecture
4. Check for memory leaks (tensors not freed)

```python
# In env config
scene.num_envs = 512  # Start small

# Check GPU memory
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### NaN Values

**Symptoms:**
```
RuntimeWarning: invalid value encountered
Training loss is NaN
```

**Diagnostic:**
```python
# Add to your environment or reward function
def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(f"  Min: {tensor.min()}, Max: {tensor.max()}")

# Check observations
check_for_nan(obs, "observations")
check_for_nan(rewards, "rewards")
```

**Common Causes:**
1. Division by zero in reward
2. Physics instability (exploding robot)
3. Log of negative number
4. Invalid actuator configuration

**Solutions:**
```python
# Safe division
reward = value / (denominator + 1e-8)

# Clamp extreme values
reward = torch.clamp(reward, -100, 100)

# Check physics params
sim_cfg = SimulationCfg(
    dt=0.005,  # Smaller timestep for stability
    substeps=2,  # More substeps
)
```

## Issue: Robot Physics Problems

### Robot Falls Through Ground

**Check 1:** Collision filters
```python
# In ArticulationCfg
spawn=sim_utils.UsdFileCfg(
    usd_path="...",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,  # Must be True
    ),
)
```

**Check 2:** Ground plane collision
```python
ground = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    ),
)
```

**Check 3:** Initial spawn height
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.5),  # Make sure above ground
)
```

### Robot Explodes/Jitters

**Cause 1:** Actuator gains too high
```python
# Bad
actuators={
    "legs": ImplicitActuatorCfg(
        stiffness=10000.0,  # Too high!
        damping=1000.0,     # Too high!
    )
}

# Better - start conservative
actuators={
    "legs": ImplicitActuatorCfg(
        stiffness=100.0,
        damping=10.0,
    )
}
```

**Cause 2:** Timestep too large
```python
# Bad
sim = SimulationCfg(dt=0.02)  # 50Hz - often unstable

# Better
sim = SimulationCfg(dt=0.005)  # 200Hz simulation
decimation = 4  # 50Hz control
```

**Cause 3:** Action scale too large
```python
# Check action bounds
actions = JointPositionActionCfg(
    scale=0.25,  # Start small, increase if robot moves too slow
)
```

### Robot Doesn't Move

**Check 1:** Actions reaching robot
```python
# Add debug print in environment
def _pre_physics_step(self, actions):
    print(f"Actions: min={actions.min():.3f}, max={actions.max():.3f}")
    super()._pre_physics_step(actions)
```

**Check 2:** Actuator configuration
```python
# Verify actuators are applied to correct joints
actuators={
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],  # Check regex
    )
}
```

**Check 3:** Joint limits
```python
# In USD or config, check limits aren't at 0
# Or position target isn't outside limits
```

## Issue: Training Problems

### Reward Not Improving

**Step 1:** Print reward values
```python
# In training loop or reward function
print(f"Reward breakdown:")
print(f"  alive: {alive_reward.mean():.4f}")
print(f"  forward: {forward_reward.mean():.4f}")
print(f"  penalty: {penalty.mean():.4f}")
print(f"  total: {total_reward.mean():.4f}")
```

**Step 2:** Check if rewards are all zeros
```python
# Common causes of zero rewards:
# 1. Threshold never reached
reward = 10.0 if distance < 0.01 else 0.0  # Too strict

# Better: shaped reward
reward = 1.0 / (1.0 + distance)

# 2. Weight is zero
rewards.success = RewTerm(func=..., weight=0.0)  # Forgot to set weight!
```

**Step 3:** Verify observation correctness
```python
# Check observations aren't constant
obs = env.reset()[0]
for i in range(10):
    next_obs, _, _, _, _ = env.step(random_action)
    diff = (next_obs - obs).abs().mean()
    print(f"Obs change: {diff:.6f}")  # Should be > 0
    obs = next_obs
```

### Training Unstable (Reward Oscillates)

**Solution 1:** Reduce learning rate
```python
algorithm = RslRlPpoAlgorithmCfg(
    learning_rate=3e-4,  # Default is 1e-3, try smaller
)
```

**Solution 2:** Normalize rewards
```python
# In reward config, ensure similar magnitudes
rewards.alive = RewTerm(func=..., weight=1.0)
rewards.forward = RewTerm(func=..., weight=1.0)  # Not 100.0
rewards.penalty = RewTerm(func=..., weight=-0.01)  # Small penalty
```

**Solution 3:** Clip rewards
```python
# In custom reward function
reward = torch.clamp(raw_reward, -10.0, 10.0)
```

### Policy Collapses (Does Nothing)

**Cause:** Entropy too low, exploration stopped

**Solution:**
```python
algorithm = RslRlPpoAlgorithmCfg(
    entropy_coef=0.01,  # Increase from default
)

policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,  # Higher initial exploration
)
```

## Debugging Commands

### Visual Debugging

```bash
# Run WITHOUT headless to see what's happening
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task MyTask-v0 \
    --num_envs 1  # Single env for clarity
```

### Check Environment Registration

```bash
# List all registered environments
./isaaclab.sh -p scripts/environments/list_envs.py | grep MyTask
```

### Verify Observations/Actions

```python
import gymnasium as gym

env = gym.make("Isaac-MyTask-v0", num_envs=1)
obs, info = env.reset()

print(f"Observation shape: {obs.shape}")
print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
print(f"Action space: {env.action_space}")

# Test random actions
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: reward={reward.item():.4f}, term={terminated.item()}")
```

### Check GPU Usage

```bash
# Monitor GPU memory and utilization
watch -n 1 nvidia-smi
```

## Common Error Messages

### "Asset not found"
```
Error: Could not find prim at path /World/Robot
```
**Solution:** Check `prim_path` matches USD structure

### "Joint not found"
```
Error: Joint 'joint1' not found in articulation
```
**Solution:** Check `joint_names_expr` regex, print available joints:
```python
print(robot.joint_names)
```

### "Fabric error"
```
Error: Fabric tensor operation failed
```
**Solution:** Usually indicates shape mismatch. Check tensor dimensions.

### "Physics material not found"
```
Warning: No physics material assigned
```
**Solution:** Add `physics_material` to spawn config

## Performance Profiling

```bash
# Run with profiler
./isaaclab.sh -p my_script.py --profiler

# Or use Python profiler
python -m cProfile -s cumulative my_script.py
```

## Checklist: Before Asking for Help

- [ ] Reproduced issue with minimal example
- [ ] Checked Isaac Lab GitHub issues
- [ ] Printed relevant tensor shapes and values
- [ ] Tried with single environment (`num_envs=1`)
- [ ] Verified without headless mode
- [ ] Checked GPU memory usage
- [ ] Reviewed recent config changes
