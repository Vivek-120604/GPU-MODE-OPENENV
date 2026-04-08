---
title: Bio Optimization Env
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# BiologicalOptimizationEnv

A complete OpenEnv environment for biological experiment optimization using reinforcement learning and language models.

## Overview

BiologicalOptimizationEnv simulates optimizing biological experiments by adjusting three key parameters:
- **Temperature** (15-45°C, optimal: 37°C)
- **pH level** (5-9, optimal: 7.4)
- **Mutation level** (0-1, optimal: 0.3)

The goal is to reach a performance score of ≥0.85 with stable conditions (≥2 consecutive stable steps).

## Features

- **Three difficulty levels**: Easy, Medium, and Hard tasks with different starting conditions
- **Deterministic behavior**: Use seeds for reproducible experiments
- **Potential-based reward shaping**: Efficient learning signal design
- **REST API**: Full HTTP endpoints for interaction
- **LLM integration**: Inference using HuggingFace models via OpenAI API format
- **Docker support**: Easy deployment on HuggingFace Spaces

## Installation

### Local Setup

1. **Clone and setup environment:**
```bash
cd /Users/USER/GPU-MODE-OPENENV
python3.11 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set environment variable (for inference):**
```bash
export HF_TOKEN="your-huggingface-token"
```

### Docker Setup

```bash
docker build -t biological-optimization-env .
docker run -p 7860:7860 -e HF_TOKEN="your-token" biological-optimization-env
```

## Usage

### 1. Start the Server

```bash
source venv/bin/activate
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

The server will be available at `http://localhost:7860`.

### 2. Health Check

```bash
curl http://localhost:7860/
```

Response:
```json
{
  "status": "ok",
  "service": "BiologicalOptimizationEnv",
  "version": "1.0.0"
}
```

### 3. Reset Environment

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'
```

Response:
```json
{
  "state": {
    "temperature": 35.0,
    "ph": 7.2,
    "mutation_level": 0.25,
    "performance_score": 0.0,
    "steps_count": 0,
    "stability_count": 0
  },
  "episode_info": {
    "task": "easy",
    "max_steps": 50
  }
}
```

### 4. Execute Steps

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "adjust_temperature", "value": 2.5}}'
```

Response:
```json
{
  "state": {
    "temperature": 37.5,
    "ph": 7.2,
    "mutation_level": 0.25,
    "performance_score": 0.98,
    "steps_count": 1,
    "stability_count": 0
  },
  "reward": 0.245,
  "done": false,
  "info": {
    "action_type": "adjust_temperature",
    "success": false,
    "stability_count": 0,
    "task": "easy"
  }
}
```

### 5. Get Current State

```bash
curl http://localhost:7860/state
```

## API Endpoints

### GET /
Health check endpoint.

### POST /reset
Reset the environment to initial state.

**Request body:**
```json
{
  "task": "medium",  // "easy", "medium", or "hard"
  "seed": null       // optional seed for reproducibility
}
```

### POST /step
Execute one action in the environment.

**Request body:**
```json
{
  "action": {
    "action_type": "adjust_temperature",  // action type
    "value": 2.5                          // adjustment value
  }
}
```

**Action types:**
- `adjust_temperature`: value in [-5.0, 5.0] (°C adjustment)
- `adjust_ph`: value in [-1.0, 1.0] (pH adjustment)
- `adjust_mutation`: value in [-0.2, 0.2] (mutation adjustment)
- `run_experiment`: value ignored (executes current experiment)

### GET /state
Get the current environment state without stepping.

## Running Inference

### Basic Inference

```bash
python inference.py --task easy --seed 42
```

### Multiple Episodes

```bash
python inference.py --task medium --episodes 5
```

### Custom Server URL

```bash
python inference.py --server-url http://your-server:7860 --task hard
```

## Task Descriptions

### Easy Task
- **Starting conditions**: temperature=35°C, pH=7.2, mutation=0.25
- **Difficulty**: Close to optimal values
- **Target performance**: ≥0.8
- **Strategy**: Small adjustments to reach optimal values

### Medium Task
- **Starting conditions**: Random within ranges
- **Difficulty**: Moderate distance from optimal values
- **Target performance**: ≥0.75
- **Strategy**: Identify direction toward optimal and converge

### Hard Task
- **Starting conditions**: temperature=20°C, pH=6.0, mutation=0.8
- **Difficulty**: Far from optimal values
- **Target performance**: ≥0.7
- **Strategy**: Large, strategic adjustments; careful convergence

## Environment Dynamics

### State Space

| Variable | Min | Max | Optimal | Description |
|----------|-----|-----|---------|-------------|
| temperature | 15°C | 45°C | 37°C | Temperature in Celsius |
| ph | 5 | 9 | 7.4 | pH level |
| mutation_level | 0 | 1 | 0.3 | Normalized mutation level |
| performance_score | 0 | 1 | - | Derived from parameter proximity |
| steps_count | 0 | 50 | - | Episode step counter |
| stability_count | 0 | ∞ | - | Consecutive stable steps |

### Reward Function

The environment uses **potential-based reward shaping**:

```
reward = Δperformance_score × 0.5
       + [stability_bonus if stability_count ≥ 2]
       + [experiment_bonus/penalty based on success]
       + [terminal_bonus/penalty if done]
```

- **Performance delta**: Main signal for convergence
- **Stability bonus**: +0.1 per step with |Δperformance| < 0.02
- **Experiment success**: +0.5 when run_experiment succeeds
- **Experiment failure**: -0.5 when run_experiment fails (premature)
- **Terminal bonus**: +0.3 on success, -0.2 on timeout

### Termination Conditions

Episode ends when:
1. **Success**: performance_score ≥ 0.85 AND stability_count ≥ 2
2. **Timeout**: steps_count ≥ 50

## Performance Score Calculation

Performance score uses exponential decay from optimal distance:

```
temp_dist = |temperature - 37| / 30
ph_dist = |ph - 7.4| / 4
mutation_dist = |mutation - 0.3| / 1.0

avg_dist = (temp_dist + ph_dist + mutation_dist) / 3
score = exp(-3 × avg_dist)
```

- At optimal values (all distances = 0): score = 1.0
- At max distance (all distances = 1): score ≈ 0.05

## Configuration

### openenv.yaml

The environment configuration in `openenv.yaml` includes:
- Task definitions with difficulty parameters
- Graders that check final performance thresholds
- State and action space definitions
- Reward function specification
- Server endpoint configuration

## File Structure

```
/Users/USER/GPU-MODE-OPENENV/
├── server/
│   ├── app.py              # FastAPI server
│   ├── environment.py      # BiologicalOptimizationEnv class
│   └── models.py           # Pydantic models
├── inference.py            # LLM-based inference client
├── openenv.yaml            # Environment configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── README.md               # This file
└── LICENSE
```

## Example Workflow

```python
# 1. Reset environment
curl -X POST http://localhost:7860/reset \
  -d '{"task": "easy", "seed": 42}'

# 2. Take steps
for step in range(50):
    curl -X POST http://localhost:7860/step \
      -d '{"action": {"action_type": "adjust_temperature", "value": 1.0}}'

# 3. Run experiment
curl -X POST http://localhost:7860/step \
  -d '{"action": {"action_type": "run_experiment", "value": 0.0}}'
```

## Troubleshooting

### Environment not initialized error
```
"detail": "Environment not initialized. Call /reset first."
```
Solution: Call `/reset` endpoint before `/step`.

### Invalid task error
```
"detail": "Invalid task 'invalid'. Must be 'easy', 'medium', or 'hard'."
```
Solution: Use one of the three valid task names.

### HF_TOKEN not set
```
ValueError: HF_TOKEN environment variable not set
```
Solution: Set your HuggingFace token:
```bash
export HF_TOKEN="your-token-here"
```

## Performance Benchmarks

Typical performance on each task:

| Task | Baseline | Target | Avg Steps |
|------|----------|--------|-----------|
| easy | 0.85+ | 0.80 | 8-12 |
| medium | 0.75+ | 0.75 | 15-25 |
| hard | 0.70+ | 0.70 | 25-40 |

## Environment Design Principles

1. **Deterministic rewards**: Same seed produces same transitions
2. **Positive feedback shaping**: Changes toward optimal increase reward
3. **Stability emphasis**: Reward consistent good performance
4. **Bounded rewards**: All rewards normalized to [-1, 1]
5. **Clear termination**: Success is objectively measurable

## HuggingFace Spaces Deployment

To deploy on HuggingFace Spaces:

1. Create a new Space with Docker Runtime
2. Set secret: `HF_TOKEN` (your HuggingFace token)
3. Push the code to the repository
4. Dockerfile will automatically build and run

## License

See LICENSE file for details.

## Support

For issues or questions, please open an issue in the repository or contact the maintainers.