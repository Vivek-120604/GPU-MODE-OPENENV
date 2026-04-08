# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptive Experimental Design environment implementation."""

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import MyAction, MyObservation, MyState
except ImportError:
    from models import MyAction, MyObservation, MyState


class MyEnvironment(Environment[MyAction, MyObservation, MyState]):
    """Deterministic environment for adaptive experimental design."""

    TASK_NAMES = {1: "easy", 2: "medium", 3: "hard"}

    TEMP_OPTIMAL = 42.0
    PH_OPTIMAL = 6.5
    MUTATION_OPTIMAL = 3

    TEMP_MIN = 20.0
    TEMP_MAX = 80.0
    PH_MIN = 4.0
    PH_MAX = 9.0
    MUTATION_MIN = 0
    MUTATION_MAX = 6

    TEMP_STEP = 1.0
    PH_STEP = 0.1

    STEP_PENALTY = 0.01
    DISTANCE_REWARD_GAIN = 2.0
    RUN_EXPERIMENT_DISTANCE_THRESHOLD = 1.0
    EARLY_OPTIMAL_SCORE = 0.99
    RUN_EXPERIMENT_QUALITY_GAIN = 0.8
    MIN_STEP_BONUS_GAIN = 0.35
    FAILED_EXPERIMENT_PENALTY = 0.6

    TASK_MAX_STEPS = {
        "easy": 6,
        "medium": 20,
        "hard": 30,
    }

    TASK_SUCCESS_CONDITIONS = {
        "easy": "Run experiment with task_score >= 0.98 within 3 steps.",
        "medium": (
            "Run experiment with task_score >= 0.90 after at least 6 steps "
            "and at least 3 distinct actions."
        ),
        "hard": (
            "Run experiment with task_score >= 0.95 after at least 20 steps "
            "and using actions 0 (increase temperature), 3 (decrease pH), "
            "and 4 (mutate sequence)."
        ),
    }

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize environment with deterministic defaults."""
        self._last_seed: int = 0
        self._active_task: str = "medium"
        self._seen_actions: set[int] = set()
        self._last_action: Optional[int] = None
        self._action_streak: int = 0
        self._episode_done: bool = False
        self._last_reward_components: dict[str, float] = {}
        self._state = self._initial_state(episode_id=str(uuid4()), task_name=self._active_task)

    def _initial_state(self, episode_id: str, task_name: str) -> MyState:
        max_steps = max(1, int(self.TASK_MAX_STEPS[task_name]))
        temperature, ph, mutation = self._task_start(task_name=task_name, seed=self._last_seed)
        distance = self._distance_to_optimal_values(temperature, ph, mutation)
        return MyState(
            episode_id=episode_id,
            step_count=0,
            temperature=temperature,
            pH=ph,
            mutation_level=mutation,
            previous_score=0.0,
            previous_distance=distance,
            steps_remaining=max_steps,
            task_name=task_name,
            task_score=self._score_from_distance(distance),
            success=False,
            success_condition=self.TASK_SUCCESS_CONDITIONS[task_name],
        )

    def _task_start(self, task_name: str, seed: int) -> tuple[float, float, int]:
        if task_name == "easy":
            # One mutation and one pH adjustment away from optimum.
            return 42.0, 6.6, 2

        if task_name == "hard":
            # Deliberately far from optimum but still solvable in 30 steps.
            return 30.0, 7.5, 6

        # Medium task: deterministic random start requiring exploration.
        rng = random.Random(seed)
        temperature = float(rng.randint(36, 48))
        ph = round(rng.uniform(5.6, 7.4), 2)
        mutation = rng.randint(0, 6)
        return temperature, ph, mutation

    def _resolve_task_name(
        self,
        task_id: Optional[int],
        task_name: Optional[str],
    ) -> str:
        if task_name is not None:
            normalized = task_name.strip().lower()
            if normalized in self.TASK_MAX_STEPS:
                return normalized
            raise ValueError(f"Unsupported task_name '{task_name}'. Use easy, medium, or hard.")

        if task_id is not None:
            if task_id in self.TASK_NAMES:
                return self.TASK_NAMES[task_id]
            raise ValueError("Unsupported task_id. Use 1 (easy), 2 (medium), or 3 (hard).")

        return "medium"

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _distance_to_optimal_values(self, temperature: float, ph: float, mutation_level: int) -> float:
        temp_error = abs(temperature - self.TEMP_OPTIMAL)
        ph_error = abs(ph - self.PH_OPTIMAL) * 10.0
        mutation_error = abs(mutation_level - self.MUTATION_OPTIMAL)
        return temp_error + ph_error + mutation_error

    def _distance_to_optimal(self) -> float:
        return self._distance_to_optimal_values(
            self._state.temperature,
            self._state.pH,
            self._state.mutation_level,
        )

    def _score_from_distance(self, distance: float) -> float:
        max_distance = 66.0
        return max(0.0, min(1.0, round(1.0 - (distance / max_distance), 4)))

    def _is_task_success(self, action_id: int) -> bool:
        if action_id != 5:
            return False

        score = self._state.task_score

        if self._active_task == "easy":
            return score >= 0.98 and self._state.step_count <= 3

        if self._active_task == "medium":
            return score >= 0.90 and self._state.step_count >= 6 and len(self._seen_actions) >= 3

        required_actions = {0, 3, 4}
        return (
            score >= 0.95
            and self._state.step_count >= 20
            and required_actions.issubset(self._seen_actions)
        )

    def _is_optimal_state(self) -> bool:
        return self._state.task_score >= self.EARLY_OPTIMAL_SCORE

    def _apply_action(self, action_id: int) -> None:
        if action_id == 0:
            self._state.temperature = self._clamp(
                self._state.temperature + self.TEMP_STEP,
                self.TEMP_MIN,
                self.TEMP_MAX,
            )
            return

        if action_id == 1:
            self._state.temperature = self._clamp(
                self._state.temperature - self.TEMP_STEP,
                self.TEMP_MIN,
                self.TEMP_MAX,
            )
            return

        if action_id == 2:
            self._state.pH = round(
                self._clamp(self._state.pH + self.PH_STEP, self.PH_MIN, self.PH_MAX),
                2,
            )
            return

        if action_id == 3:
            self._state.pH = round(
                self._clamp(self._state.pH - self.PH_STEP, self.PH_MIN, self.PH_MAX),
                2,
            )
            return

        if action_id == 4:
            # Deterministic mutation progression: cycle through fixed levels.
            next_level = self._state.mutation_level + 1
            if next_level > self.MUTATION_MAX:
                next_level = self.MUTATION_MIN
            self._state.mutation_level = next_level

    def _build_observation(self, reward: float, done: bool, action_id: int) -> MyObservation:
        return MyObservation(
            temperature=self._state.temperature,
            pH=self._state.pH,
            mutation_level=self._state.mutation_level,
            previous_score=self._state.previous_score,
            steps_remaining=self._state.steps_remaining,
            task_name=self._state.task_name,
            task_score=self._state.task_score,
            success=self._state.success,
            success_condition=self._state.success_condition,
            done=done,
            reward=reward,
            metadata={
                "action_id": action_id,
                "distance_to_optimal": self._distance_to_optimal(),
                "seed": self._last_seed,
                "reward_components": dict(self._last_reward_components),
            },
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[int] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> MyObservation:
        """
        Reset the environment.

        Returns:
            Initial deterministic observation
        """
        _ = kwargs
        self._last_seed = 0 if seed is None else seed
        self._active_task = self._resolve_task_name(task_id=task_id, task_name=task_name)
        self._seen_actions = set()
        self._last_action = None
        self._action_streak = 0
        self._episode_done = False
        self._last_reward_components = {
            "prev_distance": 0.0,
            "new_distance": 0.0,
            "delta_distance": 0.0,
            "step_penalty": 0.0,
            "run_experiment_override": 0.0,
        }
        self._state = self._initial_state(
            episode_id=episode_id or str(uuid4()),
            task_name=self._active_task,
        )
        
        self._state.success = False
        self._state.steps_remaining = max(10, int(self._state.steps_remaining))
        self._state.step_count = 0
        self._state.previous_score = 0.0
        
        print(f"DEBUG RESET: steps_remaining={self._state.steps_remaining}, success={self._state.success}")

        return self._build_observation(
            reward=0.0,
            done=False,
            action_id=-1,
        )

    def step(
        self,
        action: MyAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MyObservation:  # type: ignore[override]
        """
        Execute one deterministic control action.

        Args:
            action: MyAction containing an action_id

        Returns:
            MyObservation containing updated experimental state and reward
        """
        _ = timeout_s
        _ = kwargs

        if self._state.success or self._state.steps_remaining <= 0:
            return self._build_observation(
                reward=self._state.previous_score,
                done=True,
                action_id=action.action_id,
            )

        prev_distance = (
            abs(self._state.temperature - 42)
            + abs(self._state.pH - 6.5) * 10.0
            + abs(self._state.mutation_level - 3)
        )

        self._seen_actions.add(action.action_id)
        self._apply_action(action.action_id)

        self._state.step_count += 1
        task_max_steps = max(1, int(self.TASK_MAX_STEPS[self._active_task]))
        self._state.steps_remaining = max(task_max_steps - self._state.step_count, 0)

        new_distance = (
            abs(self._state.temperature - 42)
            + abs(self._state.pH - 6.5) * 10.0
            + abs(self._state.mutation_level - 3)
        )

        delta = prev_distance - new_distance
        if delta > 0:
            reward = 0.3 + (delta * 1.5)
        else:
            reward = -0.3 + (delta * 1.5)

        reward -= 0.002

        if not hasattr(self, "last_actions"):
            self.last_actions = []

        if (
            len(self.last_actions) >= 2
            and self.last_actions[-1] == action.action_id
            and self.last_actions[-2] == action.action_id
        ):
            reward -= 0.1

        self.last_actions.append(action.action_id)

        done = False
        if action.action_id == 5:
            if new_distance < 1.0:
                reward = 1.0
                done = True
                self._state.success = True
            else:
                reward = -0.5
                done = True
                self._state.success = False
        else:
            self._state.success = False

        if self._state.steps_remaining <= 0:
            done = True

        self._episode_done = done
        self._state.task_score = self._score_from_distance(new_distance)
        self._state.previous_distance = prev_distance

        self._last_reward_components = {
            "prev_distance": round(prev_distance, 6),
            "new_distance": round(new_distance, 6),
            "delta_distance": round(delta, 6),
            "step_penalty": 0.002,
            "run_experiment_override": round(reward, 6) if action.action_id == 5 else 0.0,
        }

        self._state.previous_score = reward
        print(f"DEBUG: prev={prev_distance:.2f}, new={new_distance:.2f}, delta={delta:.2f}, reward={reward:.2f}")

        return self._build_observation(
            reward=reward,
            done=done,
            action_id=action.action_id,
        )

    @property
    def state(self) -> MyState:
        """
        Get the current environment state.

        Returns:
            Current deterministic simulator state
        """
        return self._state
