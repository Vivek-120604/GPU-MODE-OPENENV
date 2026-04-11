"""FastAPI server for BiologicalOptimizationEnv"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import math
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

# Task configuration
TASK_CONFIG = {
    "easy":   {"threshold": 0.60, "max_steps": 35},
    "medium": {"threshold": 0.75, "max_steps": 45},
    "hard":   {"threshold": 0.85, "max_steps": 40},
}


def _compute_grade(task: str, performance_score: float) -> float:
    """Compute grade strictly in (0, 1) using sigmoid."""
    threshold = TASK_CONFIG.get(task, TASK_CONFIG["medium"])["threshold"]
    normalised = (performance_score - threshold) / max(threshold, 0.1)
    raw = 1.0 / (1.0 + math.exp(-3.0 * normalised))
    return max(0.001, min(0.999, raw))


@app.get("/", tags=["health"])
def health_check():
    return {
        "status": "ok",
        "service": "BiologicalOptimizationEnv",
        "version": "1.0.0",
    }


@app.get("/tasks", tags=["environment"])
def list_tasks():
    """Return all available tasks — required by OpenEnv spec validators."""
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Easy task with close initial conditions",
                "grader": {
                    "metric": "final_performance_score",
                    "threshold": TASK_CONFIG["easy"]["threshold"],
                    "max_steps": TASK_CONFIG["easy"]["max_steps"],
                    "score": 0.5,
                },
            },
            {
                "name": "medium",
                "description": "Medium task with moderate initial conditions",
                "grader": {
                    "metric": "final_performance_score",
                    "threshold": TASK_CONFIG["medium"]["threshold"],
                    "max_steps": TASK_CONFIG["medium"]["max_steps"],
                    "score": 0.5,
                },
            },
            {
                "name": "hard",
                "description": "Hard task with challenging initial conditions",
                "grader": {
                    "metric": "final_performance_score",
                    "threshold": TASK_CONFIG["hard"]["threshold"],
                    "max_steps": TASK_CONFIG["hard"]["max_steps"],
                    "score": 0.5,
                },
            },
        ]
    }


@app.post("/grade", tags=["environment"])
def grade(request: dict):
    """
    Grade a completed episode.
    Expected: {"task": "easy"|"medium"|"hard", "episode_result": {"performance_score": float}}
    Returns:  {"score": float}  strictly in (0, 1)
    """
    try:
        task = request.get("task", "medium")
        if task not in TASK_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown task '{task}'")

        episode_result = request.get("episode_result", {})
        performance_score = float(
            episode_result.get("performance_score",
            episode_result.get("final_performance_score", 0.0))
        )

        score = _compute_grade(task, performance_score)
        logger.info(f"Grade: task={task}, perf={performance_score:.4f}, score={score:.4f}")
        return {"score": score}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /grade: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", tags=["environment"], response_model=StateResponse)
def reset(request: ResetRequest = None):
    """Reset the environment."""
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
    """Execute one step in the environment."""
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
    """Get current environment state without stepping."""
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
