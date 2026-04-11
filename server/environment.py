
"""BiologicalOptimizationEnv - OpenEnv environment for biological experiment optimization"""
import numpy as np
from typing import Dict, Any, Tuple
from server.base import BaseEnvironment


class BiologicalOptimizationEnv(BaseEnvironment):
    """Biological Experiment Optimization Environment
    
    State space:
    - temperature: 15-45°C
    - pH: 5-9
    - mutation_level: 0-1
    - performance_score: 0-1
    - steps_count: number of steps taken
    - stability_count: consecutive stable steps
    
    Action space:
    - adjust_temperature: -5 to +5
    - adjust_ph: -1 to +1
    - adjust_mutation: -0.2 to +0.2
    - run_experiment: execute experiment
    """
    
    # Optimal values for reward calculation
    OPTIMAL_TEMP = 37.0
    OPTIMAL_PH = 7.4
    OPTIMAL_MUTATION = 0.3
    
    # State bounds
    TEMP_MIN, TEMP_MAX = 15.0, 45.0
    PH_MIN, PH_MAX = 5.0, 9.0
    MUTATION_MIN, MUTATION_MAX = 0.0, 1.0
    
    # Action bounds
    TEMP_ADJUST_RANGE = 5.0
    PH_ADJUST_RANGE = 1.0
    MUTATION_ADJUST_RANGE = 0.2
    
    def __init__(self, seed: int = None, task: str = "medium"):
        super().__init__()
        self.seed_value = seed if seed is not None else 42
        self.rng = np.random.RandomState(self.seed_value)
        self.task = task
        
        # State variables
        self.temperature = 25.0
        self.ph = 7.0
        self.mutation_level = 0.5
        self.performance_score = 0.001
        self.cumulative_reward = 0.001
        self.steps_count = 0
        self.stability_count = 0
        self.last_performance = 0.0
        self.max_steps = 50
        self.episode_info = {}
        
        self.action_counts = {
            'adjust_temperature': 0,
            'adjust_ph': 0,
            'adjust_mutation': 0
        }
        self.distinct_actions_used = set()
        self.improvement_streak = 0
        self.best_performance = 0.0
        self.last_action_type = None
        self.consecutive_same_action = 0
        self.early_convergence_step = None
        
        self._initialize_task(task)
    
    def _initialize_task(self, task: str):
        if task == "easy":
            self.temperature = self.rng.uniform(35.0, 39.0)
            self.ph = self.rng.uniform(7.2, 7.5)
            self.mutation_level = self.rng.uniform(0.20, 0.35)
            self.max_steps = 35
            
        elif task == "medium":
            temp_bias = self.rng.uniform(-8, 5) if self.rng.rand() > 0.5 else self.rng.uniform(-3, -8)
            ph_bias = self.rng.uniform(-1.2, 0.8) if self.rng.rand() > 0.5 else self.rng.uniform(-0.6, -1.2)
            self.temperature = np.clip(self.OPTIMAL_TEMP + temp_bias, self.TEMP_MIN, self.TEMP_MAX)
            self.ph = np.clip(self.OPTIMAL_PH + ph_bias, self.PH_MIN, self.PH_MAX)
            self.mutation_level = self.rng.uniform(0.05, 0.25) if self.rng.rand() > 0.5 else self.rng.uniform(0.6, 0.95)
            self.max_steps = 45
            
        elif task == "hard":
            self.temperature = self.rng.uniform(15.0, 23.0)
            self.ph = self.rng.uniform(5.0, 5.8)
            self.mutation_level = self.rng.uniform(0.75, 1.0)
            self.max_steps = 50
        
        self.steps_count = 0
        self.cumulative_reward = 0.001
        self.stability_count = 0
        self.last_performance = 0.0
        self.improvement_streak = 0
        self.best_performance = 0.0
        self.last_action_type = None
        self.consecutive_same_action = 0
        self.early_convergence_step = None
        self.action_counts = {
            'adjust_temperature': 0,
            'adjust_ph': 0,
            'adjust_mutation': 0
        }
        self.distinct_actions_used = set()
        self.performance_score = self._calculate_performance_score()
        self.episode_info = {"task": task, "max_steps": self.max_steps}
    
    def reset(self, seed: int = None, task: str = "medium") -> Dict[str, Any]:
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.RandomState(seed)
        self.task = task
        self._initialize_task(task)
        return self._get_state()
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "ph": float(self.ph),
            "mutation_level": float(self.mutation_level),
            "performance_score": float(self.performance_score),
            "steps_count": int(self.steps_count),
            "stability_count": int(self.stability_count),
        }
    
    def _calculate_performance_score(self) -> float:
        """Score based on proximity to optimal — strictly in (0.001, 0.999)."""
        temp_dist = abs(self.temperature - self.OPTIMAL_TEMP) / (self.TEMP_MAX - self.TEMP_MIN)
        ph_dist = abs(self.ph - self.OPTIMAL_PH) / (self.PH_MAX - self.PH_MIN)
        mutation_dist = abs(self.mutation_level - self.OPTIMAL_MUTATION) / (self.MUTATION_MAX - self.MUTATION_MIN)
        avg_dist = (temp_dist + ph_dist + mutation_dist) / 3.0
        score = np.exp(-3.0 * avg_dist)
        return float(max(0.001, min(0.999, score)))
    
    def _get_component_distances(self) -> Tuple[float, float, float]:
        temp_dist = abs(self.temperature - self.OPTIMAL_TEMP) / (self.TEMP_MAX - self.TEMP_MIN)
        ph_dist = abs(self.ph - self.OPTIMAL_PH) / (self.PH_MAX - self.PH_MIN)
        mutation_dist = abs(self.mutation_level - self.OPTIMAL_MUTATION) / (self.MUTATION_MAX - self.MUTATION_MIN)
        return temp_dist, ph_dist, mutation_dist
    
    def _calculate_reward(
        self,
        action_type: str,
        old_performance: float,
        new_performance: float,
        done: bool,
        success: bool,
    ) -> float:
        reward = 0.0
        perf_delta = new_performance - old_performance

        if self.task == "easy":
            step_penalty = -0.02
            perf_multiplier = 4.0
        elif self.task == "medium":
            step_penalty = -0.08
            perf_multiplier = 5.0
        else:
            step_penalty = -0.12
            perf_multiplier = 6.0

        reward += step_penalty

        # 1. Performance improvement
        if new_performance > 0.8:
            marginal_scaling = 0.5 + 0.5 * (1.0 - new_performance) / 0.2
            performance_reward = perf_delta * perf_multiplier * marginal_scaling
        else:
            performance_reward = perf_delta * perf_multiplier
        reward += performance_reward

        # 2. Oscillation penalty + diversity tracking
        # FIX: update distinct_actions_used HERE (before success check in step())
        if action_type in self.action_counts:
            if action_type == self.last_action_type:
                self.consecutive_same_action += 1
                if self.consecutive_same_action > 2:
                    repetition_penalty = -0.15 * ((self.consecutive_same_action - 2) / 3.0)
                    reward += repetition_penalty
            else:
                self.consecutive_same_action = 1
                self.distinct_actions_used.add(action_type)
            self.last_action_type = action_type
            self.action_counts[action_type] += 1

        # 3. Action diversity bonus
        total_adj = sum(self.action_counts.values())
        if self.task == "easy":
            diversity_threshold = 1
        elif self.task == "medium":
            diversity_threshold = 3
        else:
            diversity_threshold = 4

        if total_adj >= diversity_threshold:
            for action_key in self.action_counts:
                if action_key in self.distinct_actions_used:
                    ratio = self.action_counts[action_key] / max(1, total_adj)
                    if 0.2 <= ratio <= 0.5:
                        diversity_bonus = 0.15
                        reward += diversity_bonus / len(self.distinct_actions_used)

        # 4. Distance-to-optimal bonus
        distance_bonus = (1.0 - np.exp(-5.0 * new_performance)) * 0.3
        reward += distance_bonus

        # 5. Stability bonus
        if self.stability_count >= 2 and new_performance >= 0.85:
            reward += 0.15

        # 6. Improvement streak
        if perf_delta > 0.01:
            self.improvement_streak += 1
            streak_bonus = min(0.1, 0.02 + 0.01 * (self.improvement_streak - 1))
            reward += streak_bonus
        else:
            self.improvement_streak = 0

        # 7. Run experiment incentive
        if action_type == "run_experiment":
            if success:
                reward += 1.5
            elif new_performance < 0.75:
                reward -= 0.5

        # 8. Early convergence bonus
        if success and self.early_convergence_step is None:
            self.early_convergence_step = self.steps_count
            if self.task == "easy":
                speed_bonus = max(0.05, 0.2 - 0.01 * max(0, self.steps_count - 5))
            elif self.task == "medium":
                speed_bonus = max(0.1, 0.3 - 0.015 * max(0, self.steps_count - 8))
            else:
                speed_bonus = max(0.15, 0.5 - 0.02 * max(0, self.steps_count - 12))
            reward += speed_bonus

        # 9. Terminal bonus/penalty
        if done:
            if success:
                perf_quality = max(0.0, min(1.0, (new_performance - 0.85) / 0.15))
                terminal_bonus = 2.0 + 2.0 * perf_quality
                reward += terminal_bonus
            else:
                if self.task == "hard":
                    reward -= 0.8
                elif self.task == "medium":
                    reward -= 0.5
                else:
                    reward -= 0.2

        return float(reward)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        action_type = action.get("action_type", "adjust_temperature")
        value = action.get("value", 0.0)
        
        old_performance = self.performance_score

        # Task-aware noise
        if self.task == "easy":
            temp_noise, ph_noise, mut_noise = 0.12, 0.02, 0.008
        elif self.task == "medium":
            temp_noise, ph_noise, mut_noise = 0.28, 0.05, 0.02
        else:
            temp_noise, ph_noise, mut_noise = 0.60, 0.12, 0.05

        # Execute action
        if action_type == "adjust_temperature":
            delta = np.clip(value, -self.TEMP_ADJUST_RANGE, self.TEMP_ADJUST_RANGE)
            delta += self.rng.normal(0, temp_noise)
            self.temperature = np.clip(self.temperature + delta, self.TEMP_MIN, self.TEMP_MAX)
        elif action_type == "adjust_ph":
            delta = np.clip(value, -self.PH_ADJUST_RANGE, self.PH_ADJUST_RANGE)
            delta += self.rng.normal(0, ph_noise)
            self.ph = np.clip(self.ph + delta, self.PH_MIN, self.PH_MAX)
        elif action_type == "adjust_mutation":
            delta = np.clip(value, -self.MUTATION_ADJUST_RANGE, self.MUTATION_ADJUST_RANGE)
            delta += self.rng.normal(0, mut_noise)
            self.mutation_level = np.clip(self.mutation_level + delta, self.MUTATION_MIN, self.MUTATION_MAX)
        elif action_type == "run_experiment":
            self.performance_score = self._calculate_performance_score()
            self.performance_score = min(0.999, self.performance_score + self.rng.uniform(0, 0.05))

        if action_type != "run_experiment":
            self.performance_score = self._calculate_performance_score()

        # Stability check
        if abs(self.performance_score - old_performance) < 0.02:
            self.stability_count += 1
        else:
            self.stability_count = 0

        self.steps_count += 1

        # ── FIX: update distinct_actions_used BEFORE success check ──────
        # _calculate_reward() also updates it, but we need it current here.
        if action_type in self.action_counts:
            self.distinct_actions_used.add(action_type)
        # ────────────────────────────────────────────────────────────────

        # Task-aware success criteria
        distinct_count = len(self.distinct_actions_used)
        if self.task == "easy":
            success = (
                self.performance_score >= 0.85
                and self.stability_count >= 1
                and distinct_count >= 1
            )
        elif self.task == "medium":
            success = (
                self.performance_score >= 0.85
                and self.stability_count >= 2
                and distinct_count >= 2
                and self.steps_count >= 6
            )
        else:  # hard
            success = (
                self.performance_score >= 0.82
                and self.stability_count >= 2
                and distinct_count >= 3
                and self.steps_count >= 15
            )

        done = success or self.steps_count >= self.max_steps

        # Calculate reward — raw value can be outside (0,1), that's fine for training signal
        reward = self._calculate_reward(
            action_type, old_performance, self.performance_score, done, success
        )

        # FIX: clamp reward to strictly (0.001, 0.999) as required by validator
        reward = float(max(0.001, min(0.999, reward)))
        self.cumulative_reward += reward

        temp_dist, ph_dist, mutation_dist = self._get_component_distances()
        avg_distance = (temp_dist + ph_dist + mutation_dist) / 3.0
        perf_delta = self.performance_score - old_performance

        info = {
            "action_type": action_type,
            "success": success,
            "stability_count": self.stability_count,
            "task": self.task,
            "final_performance_score": float(self.performance_score),
            "total_steps": int(self.steps_count),
            # FIX: clamp cumulative_reward to (0.001, 0.999)
            "total_reward": float(max(0.001, min(0.999, self.cumulative_reward / max(1, self.steps_count)))),
            "distance_to_optimal": float(avg_distance),
            "temp_distance": float(temp_dist),
            "ph_distance": float(ph_dist),
            "mutation_distance": float(mutation_dist),
            "performance_delta": float(perf_delta),
            "performance_score": float(self.performance_score),
            "steps_taken": int(self.steps_count),
        }

        return self._get_state(), reward, done, info

    def get_state(self) -> Dict[str, Any]:
        return self._get_state()
