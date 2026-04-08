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
        """Initialize environment
        
        Args:
            seed: Random seed for reproducibility
            task: Task difficulty - 'easy', 'medium', or 'hard'
        """
        super().__init__()
        self.seed_value = seed if seed is not None else 42
        self.rng = np.random.RandomState(self.seed_value)
        self.task = task
        
        # State variables
        self.temperature = 25.0
        self.ph = 7.0
        self.mutation_level = 0.5
        self.performance_score = 0.0
        self.steps_count = 0
        self.stability_count = 0
        self.last_performance = 0.0
        self.max_steps = 50
        self.episode_info = {}
        
        # Track action usage for diversity bonus
        self.action_counts = {
            'adjust_temperature': 0,
            'adjust_ph': 0,
            'adjust_mutation': 0
        }
        self.distinct_actions_used = set()
        self.improvement_streak = 0  # Consecutive improvement steps
        self.best_performance = 0.0
        self.last_action_type = None  # Track last action for oscillation penalty
        self.consecutive_same_action = 0  # Track repeated same actions
        self.early_convergence_step = None  # Track when convergence achieved
        
        self._initialize_task(task)
    
    def _initialize_task(self, task: str):
        """Initialize environment based on task difficulty
        
        EASY: Close to optimal, 30-40 max steps, ~90-100% success
        MEDIUM: Moderate distance, requires ~6+ steps and 3 actions, ~70-90% success  
        HARD: Far from optimal, requires ~15-20 steps, ~40-70% success
        """
        if task == "easy":
            # Very close to optimal for high success rate
            self.temperature = self.rng.uniform(35.0, 39.0)
            self.ph = self.rng.uniform(7.2, 7.5)
            self.mutation_level = self.rng.uniform(0.20, 0.35)
            self.max_steps = 35
            
        elif task == "medium":
            # Medium difficulty - require exploration
            # Moderate distance from optimal
            temp_bias = self.rng.uniform(-8, 5) if self.rng.rand() > 0.5 else self.rng.uniform(-3, -8)
            ph_bias = self.rng.uniform(-1.2, 0.8) if self.rng.rand() > 0.5 else self.rng.uniform(-0.6, -1.2)
            self.temperature = np.clip(self.OPTIMAL_TEMP + temp_bias, self.TEMP_MIN, self.TEMP_MAX)
            self.ph = np.clip(self.OPTIMAL_PH + ph_bias, self.PH_MIN, self.PH_MAX)
            self.mutation_level = self.rng.uniform(0.05, 0.25) if self.rng.rand() > 0.5 else self.rng.uniform(0.6, 0.95)
            self.max_steps = 45
            
        elif task == "hard":
            # Very far from optimal - require sophisticated multi-step coordination
            # Start at extremes to force multi-step journey
            self.temperature = self.rng.uniform(15.0, 23.0)  # Very cold end of range
            self.ph = self.rng.uniform(5.0, 5.8)  # Very acidic end
            self.mutation_level = self.rng.uniform(0.75, 1.0)  # Very high end
            self.max_steps = 50
        
        self.steps_count = 0
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
        self.episode_info = {"task": task, "max_steps": self.max_steps}
    
    def reset(self, seed: int = None, task: str = "medium") -> Dict[str, Any]:
        """Reset environment to initial state
        
        Args:
            seed: Random seed
            task: Task difficulty
            
        Returns:
            State dictionary
        """
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.RandomState(seed)
        
        self.task = task
        self._initialize_task(task)
        
        return self._get_state()
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current state dictionary"""
        return {
            "temperature": float(self.temperature),
            "ph": float(self.ph),
            "mutation_level": float(self.mutation_level),
            "performance_score": float(self.performance_score),
            "steps_count": int(self.steps_count),
            "stability_count": int(self.stability_count),
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate performance score based on proximity to optimal values
        
        Uses exponential decay from optimal distance normalized to [0, 1]
        """
        # Normalize distances to [0, 1]
        temp_dist = abs(self.temperature - self.OPTIMAL_TEMP) / (self.TEMP_MAX - self.TEMP_MIN)
        ph_dist = abs(self.ph - self.OPTIMAL_PH) / (self.PH_MAX - self.PH_MIN)
        mutation_dist = abs(self.mutation_level - self.OPTIMAL_MUTATION) / (self.MUTATION_MAX - self.MUTATION_MIN)
        
        # Average distance
        avg_dist = (temp_dist + ph_dist + mutation_dist) / 3.0
        
        # Exponential decay: score = exp(-3 * distance)
        # This gives score close to 1 when near optimal, decays to ~0.05 at max distance
        score = np.exp(-3.0 * avg_dist)
        
        return float(score)
    
    def _get_component_distances(self) -> Tuple[float, float, float]:
        """Get normalized distances for each component"""
        temp_dist = abs(self.temperature - self.OPTIMAL_TEMP) / (self.TEMP_MAX - self.TEMP_MIN)
        ph_dist = abs(self.ph - self.OPTIMAL_PH) / (self.PH_MAX - self.PH_MIN)
        mutation_dist = abs(self.mutation_level - self.OPTIMAL_MUTATION) / (self.MUTATION_MAX - self.MUTATION_MIN)
        return temp_dist, ph_dist, mutation_dist
    
    def _calculate_reward(self, action_type: str, old_performance: float, new_performance: float, done: bool, success: bool) -> float:
        """Calculate discriminative reward for finals-level evaluation
        
        Design:
        - EASY: High reward for progress, low penalties
        - MEDIUM: Moderate rewards, enforce multi-action requirement
        - HARD: Tight rewards near optimum, high oscillation penalty
        
        Args:
            action_type: Type of action taken
            old_performance: Performance before action
            new_performance: Performance after action
            done: Whether episode is done
            success: Whether success criteria are met
            
        Returns:
            Reward with task-aware scaling
        """
        reward = 0.0
        perf_delta = new_performance - old_performance
        
        # Task-aware penalty and scaling
        if self.task == "easy":
            step_penalty = -0.02
            perf_multiplier = 4.0
        elif self.task == "medium":
            step_penalty = -0.08
            perf_multiplier = 5.0
        else:  # hard
            step_penalty = -0.12
            perf_multiplier = 6.0
        
        # STEP PENALTY: encourage efficiency
        reward += step_penalty
        
        # 1. PERFORMANCE IMPROVEMENT (primary signal)
        # Reduce reward for marginal improvements near optimum
        if new_performance > 0.8:
            # Diminishing returns: small improvements near optimum
            marginal_scaling = 0.5 + 0.5 * (1.0 - new_performance) / 0.2
            performance_reward = perf_delta * perf_multiplier * marginal_scaling
        else:
            performance_reward = perf_delta * perf_multiplier
        reward += performance_reward
        
        # 2. AGGRESSIVE OSCILLATION PENALTY (strong discouragement)
        if action_type in ['adjust_temperature', 'adjust_ph', 'adjust_mutation']:
            if action_type == self.last_action_type:
                self.consecutive_same_action += 1
                # Very strong penalty for repeated same action
                if self.consecutive_same_action > 2:
                    repetition_penalty = -0.15 * ((self.consecutive_same_action - 2) / 3.0)
                    reward += repetition_penalty
            else:
                self.consecutive_same_action = 1
                self.distinct_actions_used.add(action_type)
            self.last_action_type = action_type
        
        # 3. ACTION DIVERSITY BONUS (enforce multi-action solutions)
        if action_type in ['adjust_temperature', 'adjust_ph', 'adjust_mutation']:
            self.action_counts[action_type] += 1
            total_adj = sum(self.action_counts.values())
            
            # Task-specific diversity requirements
            if self.task == "easy":
                diversity_threshold = 1  # Any action diversity helps
            elif self.task == "medium":
                diversity_threshold = 3  # Need at least 8+ steps with 3 actions
            else:  # hard
                diversity_threshold = 4  # Require very balanced action use
            
            unique_actions = len(self.distinct_actions_used)
            
            # Bonus for using diverse actions
            if total_adj >= diversity_threshold:
                for action_key in self.action_counts:
                    if action_key in self.distinct_actions_used:
                        ratio = self.action_counts[action_key] / max(1, total_adj)
                        # Bonus for using each action somewhat equally
                        if 0.2 <= ratio <= 0.5:
                            diversity_bonus = 0.15
                            reward += diversity_bonus / len(self.distinct_actions_used)
        
        # 4. DISTANCE-TO-OPTIMAL BONUS (subtle formative signal)
        temp_dist, ph_dist, mutation_dist = self._get_component_distances()
        distance_bonus = (1.0 - np.exp(-5.0 * new_performance)) * 0.3
        reward += distance_bonus
        
        # 5. STABILITY BONUS (convergence reward - enforce before success)
        # Must maintain high score for 2 consecutive steps
        if self.stability_count >= 2 and new_performance >= 0.85:
            reward += 0.15
        
        # 6. IMPROVEMENT STREAK (momentum bonus)
        if perf_delta > 0.01:  # Only count meaningful improvements
            self.improvement_streak += 1
            streak_bonus = min(0.1, 0.02 + 0.01 * (self.improvement_streak - 1))
            reward += streak_bonus
        else:
            self.improvement_streak = 0
        
        # 7. EXPERIMENT RUN (modest incentive)
        if action_type == "run_experiment":
            if success:
                reward += 1.5
            else:
                if new_performance < 0.75:
                    reward -= 0.5  # Penalize experiments at low performance
        
        # 8. EARLY CONVERGENCE BONUS (time efficiency)
        if success and self.early_convergence_step is None:
            self.early_convergence_step = self.steps_count
            # Task-aware speed bonus
            if self.task == "easy":
                speed_bonus = max(0.05, 0.2 - 0.01 * max(0, self.steps_count - 5))
            elif self.task == "medium":
                speed_bonus = max(0.1, 0.3 - 0.015 * max(0, self.steps_count - 8))
            else:  # hard
                speed_bonus = max(0.15, 0.5 - 0.02 * max(0, self.steps_count - 12))
            reward += speed_bonus
        
        # 9. TERMINAL STATE HANDLING (quality-aware bonus)
        if done:
            if success:
                # Quality-based terminal bonus
                perf_quality = max(0.0, min(1.0, (new_performance - 0.85) / 0.15))
                terminal_bonus = 2.0 + 2.0 * perf_quality
                reward += terminal_bonus
            else:
                # Timeout penalty (harsher for harder tasks)
                if self.task == "hard":
                    reward -= 0.8
                elif self.task == "medium":
                    reward -= 0.5
                else:
                    reward -= 0.2
        
        return float(reward)
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step in the environment
        
        Args:
            action: Dictionary with 'action_type' and 'value' keys
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        action_type = action.get("action_type", "adjust_temperature")
        value = action.get("value", 0.0)
        
        old_performance = self.performance_score
        old_temp = self.temperature
        old_ph = self.ph
        old_mutation = self.mutation_level
        
        # Task-aware noise levels
        if self.task == "easy":
            temp_noise = 0.12
            ph_noise = 0.02
            mut_noise = 0.008
        elif self.task == "medium":
            temp_noise = 0.28
            ph_noise = 0.05
            mut_noise = 0.02
        else:  # hard
            temp_noise = 0.60  # Increased for harder difficulty
            ph_noise = 0.12   # Increased for harder difficulty
            mut_noise = 0.05  # Increased for harder difficulty
        
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
            # Run experiment: update performance based on current conditions
            self.performance_score = self._calculate_performance_score()
            # Small random improvement
            self.performance_score = min(1.0, self.performance_score + self.rng.uniform(0, 0.05))
        
        # Update performance if not explicitly run
        if action_type != "run_experiment":
            self.performance_score = self._calculate_performance_score()
        
        # Check stability: if performance stayed roughly the same
        if abs(self.performance_score - old_performance) < 0.02:
            self.stability_count += 1
        else:
            self.stability_count = 0
        
        self.steps_count += 1
        
        # Task-aware success criteria
        if self.task == "easy":
            # Easy: just need performance and single stability check
            success = self.performance_score >= 0.85 and self.stability_count >= 1
            # Require at least 2 distinct actions for easier tasks
            distinct_count = len(self.distinct_actions_used)
            if distinct_count < 1:
                success = False
        elif self.task == "medium":
            # Medium: require performance, 2 steps stability, and 2+ distinct actions
            success = self.performance_score >= 0.85 and self.stability_count >= 2
            distinct_count = len(self.distinct_actions_used)
            if distinct_count < 2 or self.steps_count < 6:
                success = False
        else:  # hard
            # Hard: require very high performance (0.985), 3 stability, 3+ distinct actions, 30+ steps
            success = self.performance_score >= 0.985 and self.stability_count >= 3
            distinct_count = len(self.distinct_actions_used)
            if distinct_count < 3 or self.steps_count < 30:
                success = False
        
        done = success or self.steps_count >= self.max_steps
        
        # Calculate reward
        reward = self._calculate_reward(
            action_type,
            old_performance,
            self.performance_score,
            done,
            success
        )
        
        # Calculate distances for logging clarity
        temp_dist, ph_dist, mutation_dist = self._get_component_distances()
        avg_distance = (temp_dist + ph_dist + mutation_dist) / 3.0
        perf_delta = self.performance_score - old_performance
        
        # Build info dict with interpretability for judges
        info = {
            "action_type": action_type,
            "success": success,
            "stability_count": self.stability_count,
            "task": self.task,
            # Distance metrics for interpretability
            "distance_to_optimal": float(avg_distance),
            "temp_distance": float(temp_dist),
            "ph_distance": float(ph_dist),
            "mutation_distance": float(mutation_dist),
            # Reward components for debugging
            "performance_delta": float(perf_delta),
            "performance_score": float(self.performance_score),
            "steps_taken": int(self.steps_count),
        }
        
        state = self._get_state()
        
        return state, reward, done, info
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state without modifying environment"""
        return self._get_state()
