"""FastAPI server for BiologicalOptimizationEnv"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional

from server.environment import BiologicalOptimizationEnv
from server.models import (
    State,
    Action,
    Observation,
    ResetRequest,
    StepRequest,
    StateResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BiologicalOptimizationEnv",
    description="OpenEnv environment for biological experiment optimization",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env: Optional[BiologicalOptimizationEnv] = None

# Task configuration — thresholds and score ranges strictly in (0, 1)
TASK_CONFIG = {
    "easy":   {"threshold": 0.60, "max_steps": 50},
    "medium": {"threshold": 0.75, "max_steps": 45},
    "hard":   {"threshold": 0.85, "max_steps": 40},
}


def _compute_grade(task: str, performance_score: float) -> float:
    """
    Compute a grade strictly in (0, 1) — never exactly 0.0 or 1.0.
    
    Formula: sigmoid-like mapping so score is always in (0.001, 0.999).
    """
    cfg = TASK_CONFIG.get(task, TASK_CONFIG["medium"])
    threshold = cfg["threshold"]

    # Normalise: how far above/below threshold (range roughly -2 to +2)
    normalised = (performance_score - threshold) / max(threshold, 0.1)

    # Sigmoid maps any real number to (0, 1) — never touches the bounds
    import math
    raw = 1.0 / (1.0 + math.exp(-3.0 * normalised))

    # Extra safety clamp: keep strictly inside (0.001, 0.999)
    return max(0.001, min(0.999, raw))


@app.get("/", tags=["health"])
def health_check():
    """Health check endpoint — must return 200 for HF Space ping"""
    return {
        "status": "ok",
        "service": "BiologicalOptimizationEnv",
        "version": "1.0.0",
    }


@app.get("/tasks", tags=["environment"])
def list_tasks():
    """Return all available tasks with grader definitions — required by OpenEnv spec"""
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Easy task with close initial conditions",
                "grader": {
                    "metric": "final_performance_score",
                    "threshold": TASK_CONFIG["easy"]["threshold"],
                    "max_steps": TASK_CONFIG["easy"]["max_steps"],
                },
            },
            {
                "name": "medium",
                "description": "Medium task with moderate initial conditions",
                "grader": {
                    "metric": "final_performance_score",
                    "threshold": TASK_CONFIG["medium"]["threshold"],
                    "max_steps": TASK_CONFIG["medium"]["max_steps"],
                },
            },
            {
                "name": "hard",
                "description": "Hard task with challenging initial conditions",
                "grader": {
                    "metric": "final_performance_score",
                    "threshold": TASK_CONFIG["hard"]["threshold"],
                    "max_steps": TASK_CONFIG["hard"]["max_steps"],
                },
            },
        ]
    }


@app.post("/grade", tags=["environment"])
def grade(request: dict):
    """
    Grade a completed episode.

    Expected request body:
        {
            "task": "easy" | "medium" | "hard",
            "episode_result": {
                "performance_score": float,   # final performance score
                "success": bool,
                "steps": int
            }
        }

    Returns:
        {"score": float}   where score is strictly in (0, 1)
    """
    try:
        task = request.get("task", "medium")
        if task not in TASK_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown task '{task}'")

        episode_result = request.get("episode_result", {})

        # Accept performance_score from episode_result or fall back to state
        performance_score = float(
            episode_result.get("performance_score",
            episode_result.get("final_performance_score", 0.0))
        )

        score = _compute_grade(task, performance_score)

        logger.info(f"Grade computed: task={task}, perf={performance_score:.4f}, score={score:.4f}")
        return {"score": score}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /grade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", tags=["environment"], response_model=StateResponse)
def reset(request: ResetRequest = None):
    """Reset the environment"""
    global env

    try:
        if request is None:
            request = ResetRequest()

        if request.task not in TASK_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task '{request.task}'. Must be one of {list(TASK_CONFIG)}.",
            )

        if env is None:
            env = BiologicalOptimizationEnv(seed=request.seed, task=request.task)
        else:
            env.reset(seed=request.seed, task=request.task)

        state = env.get_state()
        logger.info(f"Environment reset: task={request.task}, seed={request.seed}")

        return StateResponse(
            state=State(**state),
            episode_info=env.episode_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /reset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", tags=["environment"], response_model=Observation)
def step(request: StepRequest):
    """Execute one step in the environment"""
    global env

    try:
        if env is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call /reset first.",
            )

        action_dict = {
            "action_type": request.action.action_type,
            "value": request.action.value,
        }

        state, reward, done, info = env.step(action_dict)

        logger.info(
            f"Step: action={request.action.action_type}, "
            f"reward={reward:.3f}, done={done}, steps={state['steps_count']}"
        )

        return Observation(
            state=State(**state),
            reward=reward,
            done=done,
            info=info,
        )

    except Exception as e:
        logger.error(f"Error in /step: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", tags=["environment"], response_model=StateResponse)
def get_state():
    """Get current environment state without stepping"""
    global env

    try:
        if env is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call /reset first.",
            )

        state = env.get_state()
        return StateResponse(
            state=State(**state),
            episode_info=env.episode_info,
        )

    except Exception as e:
        logger.error(f"Error in /state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    logger.info("Starting BiologicalOptimizationEnv server...")
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
