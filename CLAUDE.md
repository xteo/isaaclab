# Isaac Lab

GPU-accelerated robot learning framework built on NVIDIA Isaac Sim.

## Quick Commands

```bash
# Run any script with Isaac Sim
./isaaclab.sh -p <script.py> [args]

# Training (RSL-RL)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK_NAME>

# Play trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task <TASK_NAME>

# List available environments
./isaaclab.sh -p scripts/environments/list_envs.py

# Convert URDF to USD
./isaaclab.sh -p scripts/tools/convert_urdf.py <input.urdf> <output.usd>
```

## Key Locations

| What | Where |
|------|-------|
| Environments | `/source/isaaclab_tasks/isaaclab_tasks/manager_based/` |
| Robot configs | `/source/isaaclab_assets/isaaclab_assets/robots/` |
| Core framework | `/source/isaaclab/isaaclab/` |
| MDP components | `/source/isaaclab/isaaclab/envs/mdp/` |
| Tutorials | `/scripts/tutorials/` |
| Training scripts | `/scripts/reinforcement_learning/` |

## Environment Development

1. **Manager-based envs** (recommended): Modular, multi-file, use managers for observations/actions/rewards
2. **Direct envs**: Single-file, explicit control loop

Pattern: Copy from `/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/` and modify.

## Configuration Pattern

All configs use `@configclass` decorator. Key classes:
- `ManagerBasedRLEnvCfg` - RL environment config
- `ArticulationCfg` - Robot asset config
- `ObservationGroupCfg`, `ActionGroupCfg`, `RewardTermCfg` - MDP components

## Detailed Documentation

See `.claude/docs/` for comprehensive guides:
- `REPOSITORY_STRUCTURE.md` - Full architecture and navigation
- `AGENT_STRATEGY.md` - Strategy for AI-assisted development
- `RESEARCH_OVERVIEW.md` - Related tools and practices
