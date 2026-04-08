# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the adaptive experimental design environment."""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class MyAction(Action):
    """Discrete action for adaptive experimental design."""

    action_id: int = Field(
        ...,
        ge=0,
        le=5,
        description=(
            "Action ID: 0=increase temperature, 1=decrease temperature, "
            "2=increase pH, 3=decrease pH, 4=mutate sequence, 5=run experiment"
        ),
    )


class MyObservation(Observation):
    """Observation after each control action or experiment run."""

    temperature: float = Field(description="Current temperature")
    pH: float = Field(description="Current pH")
    mutation_level: int = Field(description="Current mutation level")
    previous_score: float = Field(description="Previous step reward")
    steps_remaining: int = Field(description="Remaining decision steps")
    task_name: str = Field(description="Active task name: easy, medium, or hard")
    task_score: float = Field(ge=0.0, le=1.0, description="Task score in [0, 1]")
    success: bool = Field(description="Whether the task success condition is satisfied")
    success_condition: str = Field(description="Human-readable success condition")
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: float = Field(default=0.0, description="Reward received for the last action")
    metadata: dict = Field(default_factory=dict, description="Extra debug info")


class MyState(State):
    """Internal simulator state exposed through /state."""

    temperature: float = Field(description="Current temperature")
    pH: float = Field(description="Current pH")
    mutation_level: int = Field(description="Current mutation level")
    previous_score: float = Field(description="Previous step reward")
    previous_distance: float = Field(description="Distance to optimal state before latest action")
    steps_remaining: int = Field(description="Remaining decision steps")
    task_name: str = Field(description="Active task name: easy, medium, or hard")
    task_score: float = Field(ge=0.0, le=1.0, description="Task score in [0, 1]")
    success: bool = Field(description="Whether the task success condition is satisfied")
    success_condition: str = Field(description="Human-readable success condition")
