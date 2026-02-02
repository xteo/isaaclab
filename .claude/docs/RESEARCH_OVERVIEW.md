# Research Overview: AI Agents for Robotics Simulation

> A comprehensive compilation of similar practices, tools, and approaches for AI-assisted robotics simulation development.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [LLM-Based Reward Generation](#llm-based-reward-generation)
3. [Code Generation for Robotics](#code-generation-for-robotics)
4. [AI-Friendly Repository Practices](#ai-friendly-repository-practices)
5. [MuJoCo Ecosystem & Translation](#mujoco-ecosystem--translation)
6. [Simulation Copilots & Assistants](#simulation-copilots--assistants)
7. [Vision-Language-Action Models](#vision-language-action-models)
8. [Key Takeaways & Recommendations](#key-takeaways--recommendations)

---

## Executive Summary

The intersection of LLMs and robotics simulation is rapidly evolving. This research identified several key trends:

1. **Automatic Reward Generation** - EUREKA, Text2Reward, and DrEureka demonstrate that LLMs can generate reward functions that match or exceed human-designed ones
2. **Code as Interface** - Programming language structures significantly improve LLM output for robotics tasks
3. **Simulation Acceleration** - GPU-accelerated simulators (Isaac Lab, MuJoCo MJX, Genesis) enable rapid iteration
4. **Emerging Standards** - AGENTS.md and llms.txt are becoming standards for AI-friendly repositories
5. **Skills Pattern** - Claude Code's skills system provides a model for modular, context-aware assistance

---

## LLM-Based Reward Generation

### EUREKA (NVIDIA, ICLR 2024)

**Overview:** Automatic reward function generation using LLMs with evolutionary optimization.

**Key Innovation:**
- Takes environment source code + task description as input
- Zero-shot generates executable reward functions
- Uses RL training feedback to iteratively improve rewards

**Results:**
- Outperforms human-written rewards on 83% of 29 tasks
- 52% average improvement over expert designs
- Enabled novel behaviors (pen spinning) previously unachievable

**Architecture:**
```
Task Description + Env Code → LLM → Candidate Rewards
                                         ↓
                               GPU-Accelerated RL Training
                                         ↓
                               Training Statistics Feedback
                                         ↓
                               LLM Refinement (repeat)
```

**Repository:** https://github.com/eureka-research/Eureka

**Isaac Lab Integration:** https://github.com/isaac-sim/IsaacLabEureka

### DrEureka (RSS 2024)

**Extension:** Adds automatic domain randomization to EUREKA.

**Key Features:**
- Reward-Aware Physics Priors (RAPP)
- Automates sim-to-real transfer parameters
- Zero-shot sim-to-real deployment

**Results:**
- 34% faster locomotion than human designs
- 300% more in-hand rotations
- Novel tasks: quadruped yoga ball balancing

**Repository:** https://github.com/eureka-research/DrEureka

### Text2Reward (ICLR 2024 Spotlight)

**Approach:** Dense reward function generation from natural language.

**Key Features:**
- Interpretable, free-form reward code
- Iterative refinement with human feedback
- Handles complex manipulation tasks

**Results:**
- Similar or better than expert rewards on 13/17 tasks
- Successful on real robot deployment

**Repository:** https://github.com/xlang-ai/text2reward

### Comparison Table

| System | Input | Output | Feedback Loop | Real Robot |
|--------|-------|--------|---------------|------------|
| EUREKA | Code + NL | Reward function | RL statistics | No |
| DrEureka | Code + NL | Reward + DR | RL statistics | Yes |
| Text2Reward | NL only | Reward function | Human | Yes |
| CurricuLLM | NL | Curriculum + Rewards | RL statistics | Yes |

---

## Code Generation for Robotics

### Code as Policies (Google)

**Concept:** LLMs write executable robot policy code from natural language.

**Key Innovations:**
- Hierarchical code generation (recursively defines functions)
- Supports reactive policies and impedance controllers
- Vision-based pick-and-place capabilities

**Results:**
- 90%+ success on seen instructions
- Generalizes to unseen object compositions

**Example:**
```python
# Natural language: "Put the red block on top of the blue block"
# Generated code:
def task():
    red_block_pos = detect_object("red block")
    blue_block_pos = detect_object("blue block")
    grasp(red_block_pos)
    place(blue_block_pos + [0, 0, 0.05])  # Stack offset
```

**Website:** https://code-as-policies.github.io/

### ProgPrompt

**Approach:** Use programming language structures for task planning.

**Key Features:**
- Pythonic program headers with available actions
- Plans include comments, actions, and assertions
- Evaluated in VirtualHome environment

**Example Prompt Structure:**
```python
# Available actions: pick(obj), place(obj, location), open(obj)
# Available objects: apple, bowl, fridge
# Task: Put the apple in the bowl

def task():
    pick("apple")
    place("apple", "bowl")
```

**Website:** https://progprompt.github.io/

### GenSim / GenSim2

**Capability:** Generate simulation tasks from natural language.

**Features:**
- Goal-directed and exploratory generation modes
- Expanded benchmark from 10 to 100+ tasks
- 25% improvement on real-world transfer

**Repository:** https://github.com/liruiw/GenSim

### RoboGen

**Capability:** Self-guided agent for continuous skill acquisition.

**Features:**
- Autonomously proposes new tasks
- Generates corresponding environments
- Learns skills without human intervention

**Repository:** https://github.com/Genesis-Embodied-AI/RoboGen

---

## AI-Friendly Repository Practices

### AGENTS.md Standard

**Overview:** Open standard backed by Google, OpenAI, Cursor, and Linux Foundation.

**Adoption:** 60,000+ repositories

**Key Features:**
- Distributed documentation (per-directory files)
- Nearest-file takes precedence
- Six core content areas

**Recommended Structure:**
```markdown
# Project Name

## Commands
- `npm run build` - Build the project
- `npm test` - Run tests

## Project Structure
- `/src/components` - React components
- `/src/api` - API client code

## Code Style
- Use TypeScript strict mode
- Prefer named exports

## Boundaries
- Never commit secrets
- Never modify /vendor
```

**Website:** https://agents.md/

### CLAUDE.md Pattern

**Purpose:** Claude Code-specific configuration file.

**Best Practices:**
- Keep concise (<60 lines)
- Prefer pointers to copies
- Focus on universal rules
- Use progressive disclosure

**Example:**
```markdown
# Isaac Lab

## Quick Commands
./isaaclab.sh -p script.py  # Run with Isaac Sim

## Key Locations
- Environments: /source/isaaclab_tasks/
- Robots: /source/isaaclab_assets/robots/
- Tutorials: /scripts/tutorials/

## Conventions
- Use @configclass for all configs
- Register envs with gymnasium
```

### Claude Code Skills

**Structure:**
```
.claude/skills/
├── skill-name/
│   ├── SKILL.md          # Required: frontmatter + instructions
│   └── templates/        # Optional: code templates
```

**SKILL.md Format:**
```yaml
---
name: skill-name
description: Clear description of what and when to use
---
# Instructions
Detailed instructions loaded on-demand
```

**Key Principles:**
- Progressive disclosure (load on trigger)
- Description is discovery mechanism
- Keep body under 500 lines
- Include templates for code generation

### llms.txt Standard

**Purpose:** Standardized file for LLM access to website content.

**Files:**
- `/llms.txt` - Index with links and descriptions
- `/llms-full.txt` - All content in single file

**Adoption:** 844,000+ websites

**Website:** https://llmstxt.org/

---

## MuJoCo Ecosystem & Translation

### Key MuJoCo Resources

| Resource | Purpose | Link |
|----------|---------|------|
| MuJoCo Menagerie | High-quality robot models | github.com/google-deepmind/mujoco_menagerie |
| dm_control | Benchmark environments | github.com/google-deepmind/dm_control |
| MuJoCo Playground | GPU-accelerated (MJX) | playground.mujoco.org |
| LocoMuJoCo | Locomotion IL | github.com/robfiras/loco-mujoco |

### MuJoCo → Isaac Lab Translation

**mjlab Bridge:** https://github.com/mujocolab/mjlab
- Combines Isaac Lab API with MuJoCo physics
- Migration guide included

**Conceptual Mapping:**

| MuJoCo | Isaac Lab |
|--------|-----------|
| `gym.Env` | `ManagerBasedRLEnv` |
| `_get_obs()` | `ObservationManager` |
| `step(action)` | `ActionManager` |
| `_get_reward()` | `RewardManager` |
| `reset()` | `EventManager` |
| Domain randomization | `EventManager` modes |

### Imitation Learning Frameworks

**robomimic:**
- Primary IL framework for MuJoCo
- Algorithms: BC, BC-RNN, Diffusion Policy
- Isaac Lab integration: `/scripts/imitation_learning/robomimic/`

**D4RL:**
- Offline RL datasets
- Standardized benchmarks
- Convert to robomimic format available

### Performance Comparison

| Simulator | Speed | Key Feature |
|-----------|-------|-------------|
| Isaac Lab | 82-94K FPS (4096 envs) | Omniverse integration |
| MuJoCo MJX | 2.7M steps/sec (TPU) | TPU optimization |
| Genesis | 43M FPS | Multi-physics |
| ManiSkill3 | 30K+ FPS | Visual-state |

---

## Simulation Copilots & Assistants

### Existing Commercial Solutions

**Siemens Process Simulate Copilot:**
- AI assistant in Process Simulate X
- Robot path optimization
- Collision resolution
- Real-time guidance

**InOrbit RobOps Copilot:**
- Fleet management AI
- Incident analysis
- Performance optimization
- Natural language queries

**fruitcore robotics AI Copilot:**
- ChatGPT-powered robot programming
- Function and template generation
- Real-time debugging

### Research Prototypes

**FAEA (Claude Agent SDK):**
- Direct embodied manipulation
- 84.9% LIBERO success
- No demonstrations required

**Language to Rewards:**
- MuJoCo MPC integration
- Natural language to rewards
- Open-source implementation

**Prompt2Walk:**
- LLM outputs joint positions
- Few-shot prompting
- Quadruped locomotion

---

## Vision-Language-Action Models

### RT-2 (Google)

**Architecture:** Actions as text tokens

**Models:** PaLI-X (5B-55B), PaLM-E (12B)

**Features:**
- Chain-of-thought reasoning
- Multi-stage task planning
- Web-scale pretraining benefits

### OpenVLA

**Stats:** 7B parameters, open-source

**Training:** 970K real-world demonstrations

**Results:** Outperforms RT-2-X (55B) by 16.5%

**Repository:** https://github.com/openvla/openvla

### VoxPoser

**Approach:** Zero-shot trajectory synthesis

**Components:**
- OWL-ViT, SAM, XMEM for perception
- Voxel-space reasoning
- Motion planner integration

**Repository:** https://github.com/huangwl18/VoxPoser

---

## Key Takeaways & Recommendations

### For Isaac Lab Agent Development

1. **Adopt EUREKA Pattern**
   - Use environment source code as context
   - Iterate with training feedback
   - Support human refinement

2. **Implement Skills System**
   - Modular, loadable knowledge units
   - Progressive disclosure
   - Template-based code generation

3. **Create Translation Guides**
   - Explicit MuJoCo → Isaac Lab mappings
   - Side-by-side code examples
   - Physics parameter equivalents

4. **Build Debugging Intelligence**
   - Common issues database
   - Diagnostic question trees
   - Automated fix suggestions

5. **Add Repository Instrumentation**
   - CLAUDE.md at root
   - AGENTS.md for cross-tool support
   - Per-module documentation

### Repository Best Practices

| Practice | Implementation |
|----------|---------------|
| Progressive disclosure | Root pointer → detailed docs |
| Distributed documentation | Per-directory AGENTS.md |
| Template library | `/tools/templates/` |
| Code generation | Skills with template expansion |
| Validation | Pre-commit checks for configs |

### Critical Success Factors

1. **Code Quality** - Generated code must be syntactically and semantically correct
2. **Context Relevance** - Load only necessary information
3. **Feedback Loops** - Learn from training outcomes
4. **Human-in-Loop** - Support iterative refinement
5. **Cross-Framework** - Bridge MuJoCo and other ecosystems

---

## Reference Links

### Official Resources
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- MuJoCo: https://mujoco.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/

### Research Papers
- EUREKA: https://arxiv.org/abs/2310.12931
- DrEureka: https://arxiv.org/abs/2406.01967
- Code as Policies: https://arxiv.org/abs/2209.07753
- GenSim: https://arxiv.org/abs/2310.01361
- Text2Reward: https://arxiv.org/abs/2309.11489

### GitHub Repositories
- EUREKA: https://github.com/eureka-research/Eureka
- Isaac Lab Eureka: https://github.com/isaac-sim/IsaacLabEureka
- GenSim: https://github.com/liruiw/GenSim
- RoboGen: https://github.com/Genesis-Embodied-AI/RoboGen
- OpenVLA: https://github.com/openvla/openvla
- VoxPoser: https://github.com/huangwl18/VoxPoser
- Awesome LLM Robotics: https://github.com/GT-RIPL/Awesome-LLM-Robotics

### Standards & Guides
- AGENTS.md: https://agents.md/
- llms.txt: https://llmstxt.org/
- Claude Code Skills: https://code.claude.com/docs/en/skills

### Curated Lists
- Awesome Cursor Rules: https://github.com/PatrickJS/awesome-cursorrules
- LLM-RL Papers: https://github.com/WindyLab/LLM-RL-Papers
- LLM Agents Papers: https://github.com/AGI-Edgerunners/LLM-Agents-Papers

---

*This research overview should be updated as the field evolves. Last updated: February 2026.*
