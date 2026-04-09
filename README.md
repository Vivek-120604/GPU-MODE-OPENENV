---
title: Bio Optimization Env
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧬 BiologicalOptimizationEnv

<div align="center">
  <p><b>A production-grade, multi-mode compliant OpenEnv environment for biological experiment optimization using Reinforcement Learning and LLMs.</b></p>
</div>

## 🌟 Overview

BiologicalOptimizationEnv simulates the complex, high-stakes process of optimizing biological experiments. Agents must adjust three critical, continuous parameters to find the perfect homeostasis for an experiment to succeed:
- 🌡️ **Temperature** (15-45°C, optimal: 37°C)
- 🧪 **pH level** (5.0-9.0, optimal: 7.4)
- 🧬 **Mutation level** (0-1.0, optimal: 0.3)

The objective is to achieve a `performance_score` of **≥0.85** while maintaining **homeostasis** (≥2 consecutive stable steps) within a strict time budget. 

---

## 🚀 Why This Is A Tier-1 RL Testbed

Unlike simple grid-worlds, this environment is specifically designed to relentlessly evaluate an agent's ability to explore efficiently, avoid local minima, and plan multi-step trajectories.

### 🛡️ "Anti-Gaming" Mechanics
We have implemented rigorous constraints that prevent agents from "gaming" the reward system:
1. **The Oscillation Penalty**: Agents that frantically alternate between two actions (e.g., `+temp`, `-temp`, `+temp`) are hit with compounding negative rewards.
2. **Action Diversity Enforcement**: Success structurally requires the agent to utilize *multiple* adjustment levers. The Medium and Hard tasks will **fail** the agent if they only optimize a single parameter while ignoring the others.
3. **The Stability Lock (Homeostasis)**: Striking the optimal values isn't enough. The agent must *hold* the values steady for consecutive steps, proving it understands convergence rather than just getting lucky.

### 🎯 Target-Aware Inference
The provided inference agent (`inference.py`) utilizes a **Target-Aware Goal-Directed Strategy**. Instead of being trapped in reward-chasing oscillation, it mathematically calculates the largest normalized gap to the optimum and actively drives completion. 

---

## 📊 Performance Benchmarks

Our default Target-Aware Inference Agent utterly crushes the baselines, demonstrating the environment's responsiveness to highly optimized policies:

| Task | Baseline Targets | Our Agent's Steps | Our Agent's Score | Result |
|------|------------------|-------------------|-------------------|--------|
| **Easy** | 0.85+ | 3 - 5 | ~0.98 | ✅ Success |
| **Medium** | 0.75+ | **7** *(vs 25 baseline)* | **~0.97** | ✅ Success |
| **Hard** | 0.70+ | **12 - 15** | **~0.92** | ✅ Success |

---

## 🛠️ Multi-Mode Deployment Ready

This repository rigorously complies with the **OpenEnv Multi-Mode Deployment Spec**:
- ✅ Deterministic locking via `uv.lock`.
- ✅ Complete `[project.scripts]` definitions mapping to a callable `server.app:main`.
- ✅ `openenv-core>=0.2.0` native support.
- ✅ Pristine `Dockerfile` utilizing lightweight `python:3.11-slim` caching layers.

### Local Setup

1. **Clone and setup environment:**
```bash
git clone https://github.com/Vivek-120604/GPU-MODE-OPENENV.git
cd GPU-MODE-OPENENV
python3 -m venv venv
source venv/bin/activate
```

2. **Sync Dependencies:**
```bash
pip install uv
uv sync
# OR fallback to pip:
pip install -r requirements.txt
```

3. **Start the Server**
```bash
openenv-bio-server
# OR
python -m server.app
```

### Docker Deployment
```bash
docker build -t bio-env .
docker run -p 7860:7860 -e HF_TOKEN="your-token" bio-env
```

---

## 🧬 Environment Dynamics Deep Dive

### Reward Function
The environment utilizes a meticulously balanced **potential-based reward shaping** formula bounding all outputs to `[-1, 1]`:

```text
reward = (Δperformance_score × multiplier)
       - (step_penalty)
       - (oscillation_penalty if consecutive same action)
       + (diversity_bonus if distinct_actions > threshold)
       + (stability_bonus if stability_count ≥ 2)
```

### Performance Score Calculation
Performance decays exponentially based on the normalized coordinate distance from the absolute optimums:

```python
temp_dist = abs(temperature - 37) / 30
ph_dist = abs(ph - 7.4) / 4
mutation_dist = abs(mutation - 0.3) / 1.0

avg_dist = (temp_dist + ph_dist + mutation_dist) / 3
score = exp(-3 × avg_dist)
```
*At true optimal (0 distance), score = 1.0. At maximum distance, score ≈ 0.05.*

---

## 📡 API Spec

The server exposes strict, stateless REST endpoints required by the OpenEnv grading system:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Health check |
| `GET`  | `/tasks` | Retrieves registered difficulties (Easy, Medium, Hard) |
| `POST` | `/reset` | Re-initializes state space and seeds |
| `POST` | `/step`  | Dispatches bounded `adjust_*` actions, returning `Observation` |
| `GET`  | `/state` | Exposes raw current coordinates (read-only) |

---
*Developed for the Meta OpenEnv Agentic RL Hackathon.*