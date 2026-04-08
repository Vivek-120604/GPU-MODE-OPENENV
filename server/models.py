"""Pydantic models for BiologicalOptimizationEnv"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class State(BaseModel):
    """Observation state of the environment"""
    temperature: float = Field(..., description="Temperature in Celsius (15-45°C)")
    ph: float = Field(..., description="pH level (5-9)")
    mutation_level: float = Field(..., description="Mutation level (0-1)")
    performance_score: float = Field(..., description="Performance score (0-1)")
    steps_count: int = Field(..., description="Number of steps taken")
    stability_count: int = Field(..., description="Consecutive stable steps")


class Action(BaseModel):
    """Action to take in the environment"""
    action_type: str = Field(..., description="Type of action: 'adjust_temperature', 'adjust_ph', 'adjust_mutation', 'run_experiment'")
    value: float = Field(default=0.0, description="Value for the action")


class Observation(BaseModel):
    """Observation returned from environment step"""
    state: State
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    """Request to reset environment"""
    task: str = Field(default="medium", description="Task name: 'easy', 'medium', or 'hard'")
    seed: int = Field(default=None, description="Random seed for reproducibility")


class StepRequest(BaseModel):
    """Request to step environment"""
    action: Action


class StateResponse(BaseModel):
    """Response with current state"""
    state: State
    episode_info: Dict[str, Any] = Field(default_factory=dict)
