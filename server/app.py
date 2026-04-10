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
    """Return all available tasks — required by OpenEnv spec validators"""
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Easy task with close initial conditions",
                "grader": {"metric": "final_performance_score", "threshold": 0.8},
            },
            {
                "name": "medium",
                "description": "Medium task with moderate initial conditions",
                "grader": {"metric": "final_performance_score", "threshold": 0.75},
            },
            {
                "name": "hard",
                "description": "Hard task with challenging initial conditions",
                "grader": {"metric": "final_performance_score", "threshold": 0.7},
            },
        ]
    }


@app.post("/reset", tags=["environment"], response_model=StateResponse)
def reset(request: ResetRequest = None):
    """Reset the environment
    
    Args:
        request: ResetRequest with task name and optional seed
        
    Returns:
        StateResponse with initial state
    """
    global env
    
    try:
        if request is None:
            request = ResetRequest()
        
        # Validate task
        if request.task not in ["easy", "medium", "hard"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task '{request.task}'. Must be 'easy', 'medium', or 'hard'.",
            )
        
        # Create or reset environment
        if env is None:
            env = BiologicalOptimizationEnv(seed=request.seed, task=request.task)
        else:
            env.reset(seed=request.seed, task=request.task)
        
        state = env.get_state()
        
        logger.info(f"Environment reset with task={request.task}, seed={request.seed}")
        
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
    """Execute one step in the environment
    
    Args:
        request: StepRequest with action
        
    Returns:
        Observation with state, reward, done, and info
    """
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
            f"Step executed: action={request.action.action_type}, "
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
    """Get current environment state without stepping
    
    Returns:
        StateResponse with current state
    """
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
    """Main entry point for starting the environment server"""
    import uvicorn
    
    logger.info("Starting BiologicalOptimizationEnv server...")
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
        reload=False
    )


if __name__ == "__main__":
    main()
