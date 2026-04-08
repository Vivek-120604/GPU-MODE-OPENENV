# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptive Experimental Design environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import MyAction, MyObservation, MyState


class MyEnv(
    EnvClient[MyAction, MyObservation, MyState]
):
    """
    Client for the Adaptive Experimental Design environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.temperature, result.observation.pH)
        ...
        ...     result = client.step(MyAction(action_id=0))
        ...     print(result.observation.temperature)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MyEnv.from_docker_image("my_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MyAction(action_id=5))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MyAction) -> Dict:
        """
        Convert MyAction to JSON payload for step message.

        Args:
            action: MyAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_id": action.action_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
        """
        Parse server response into StepResult[MyObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MyObservation
        """
        obs_data = payload.get("observation", {})
        observation = MyObservation(
            temperature=obs_data.get("temperature", 0.0),
            pH=obs_data.get("pH", 0.0),
            mutation_level=obs_data.get("mutation_level", 0),
            previous_score=obs_data.get("previous_score", 0.0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            task_name=obs_data.get("task_name", "medium"),
            task_score=obs_data.get("task_score", 0.0),
            success=obs_data.get("success", False),
            success_condition=obs_data.get("success_condition", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> MyState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            MyState object with current experiment state
        """
        return MyState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            temperature=payload.get("temperature", 0.0),
            pH=payload.get("pH", 0.0),
            mutation_level=payload.get("mutation_level", 0),
            previous_score=payload.get("previous_score", 0.0),
            previous_distance=payload.get("previous_distance", 0.0),
            steps_remaining=payload.get("steps_remaining", 0),
            task_name=payload.get("task_name", "medium"),
            task_score=payload.get("task_score", 0.0),
            success=payload.get("success", False),
            success_condition=payload.get("success_condition", ""),
        )

print("🔥 USING CORRECT ENV FILE") 