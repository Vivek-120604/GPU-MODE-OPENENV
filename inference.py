#!/usr/bin/env python3

import os
import json
import requests
import time
import random
from typing import Dict, Any, Optional

# Try to import OpenAI if available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Environment configuration
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS = 50
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Optimal targets for biological optimization
OPTIMAL_TEMP = 37.0
OPTIMAL_PH = 7.4
OPTIMAL_MUTATION = 0.3

class ImprovedOptimizationAgent:
    def __init__(self):
        self.step_count = 0
        self.last_performance = 0.0
        self.last_reward = 0.0
        self.improvements = 0
        self.use_llm = HF_TOKEN and OPENAI_AVAILABLE
        self.action_rotation = 0  # Track action rotation for balanced selection
        self.actions = ['adjust_temperature', 'adjust_ph', 'adjust_mutation']
        self.action_priority_cycle = 0  # Cycle through action priorities
        
        if self.use_llm:
            try:
                self.llm_client = OpenAI(
                    api_key=HF_TOKEN,
                    base_url="https://router.huggingface.co/v1"
                )
            except:
                self.use_llm = False
        
    def get_llm_action(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get action from LLM if available"""
        if not self.use_llm:
            return None
            
        try:
            temp = state.get('temperature', 25.0)
            ph = state.get('ph', 7.0)
            mutation = state.get('mutation_level', 0.5)
            performance = state.get('performance_score', 0.0)
            
            prompt = f"""Given biological experiment optimization conditions:
- Temperature: {temp:.1f}°C (optimal: 37°C)
- pH: {ph:.2f} (optimal: 7.4)
- Mutation level: {mutation:.2f} (optimal: 0.3)
- Performance score: {performance:.3f}

Choose ONE action to maximize performance:
1. adjust_temperature with value in [-5.0, 5.0]
2. adjust_ph with value in [-1.0, 1.0]
3. adjust_mutation with value in [-0.2, 0.2]
4. run_experiment (if performance >= 0.85)

Return ONLY: action_type=X value=Y (or just action_type=X for run_experiment)"""

            response = self.llm_client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            if "adjust_temperature" in response_text:
                try:
                    value = float(response_text.split("value=")[1].split()[0])
                    return {'action_type': 'adjust_temperature', 'value': float(max(-5.0, min(5.0, value)))}
                except:
                    pass
            elif "adjust_ph" in response_text:
                try:
                    value = float(response_text.split("value=")[1].split()[0])
                    return {'action_type': 'adjust_ph', 'value': float(max(-1.0, min(1.0, value)))}
                except:
                    pass
            elif "adjust_mutation" in response_text:
                try:
                    value = float(response_text.split("value=")[1].split()[0])
                    return {'action_type': 'adjust_mutation', 'value': float(max(-0.2, min(0.2, value)))}
                except:
                    pass
            elif "run_experiment" in response_text:
                return {'action_type': 'run_experiment', 'value': 0.0}
                
        except:
            pass
        
        return None
        
    def get_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Balanced optimization with action prioritization rotation.
        Systematically cycles action priorities while using performance-driven selection.
        Eliminates bias by ensuring each action gets priority periodically.
        """
        self.step_count += 1
        
        temp = state.get('temperature', 25.0)
        ph = state.get('ph', 7.0)
        mutation = state.get('mutation_level', 0.5)
        performance = state.get('performance_score', 0.0)
        stability = state.get('stability_count', 0)
        
        # Track improvements
        if performance > self.last_performance:
            self.improvements += 1
        self.last_performance = performance
        
        # Try LLM first (15% of the time for intelligent steering)
        if self.use_llm and random.random() < 0.15:
            llm_action = self.get_llm_action(state)
            if llm_action:
                return llm_action
        
        # Check if we can run experiment (performance >= 0.85 and stable >= 2)
        if performance >= 0.85 and stability >= 2:
            return {'action_type': 'run_experiment', 'value': 0.0}
        
        # Calculate distances from optimal
        temp_diff = OPTIMAL_TEMP - temp
        ph_diff = OPTIMAL_PH - ph
        mutation_diff = OPTIMAL_MUTATION - mutation
        
        temp_distance = abs(temp_diff)
        ph_distance = abs(ph_diff)
        mutation_distance = abs(mutation_diff)
        
        # Adaptive step sizing - reduce step size as we get closer
        temp_step_size = min(5.0, max(0.5, temp_distance * 0.35))
        ph_step_size = min(1.0, max(0.1, ph_distance * 0.35))
        mutation_step_size = min(0.2, max(0.02, mutation_distance * 0.35))
        
        # Action distances for prioritization
        action_distances = {
            'adjust_temperature': temp_distance,
            'adjust_ph': ph_distance,
            'adjust_mutation': mutation_distance
        }
        
        # Create priority order: cycle through actions to ensure fairness
        priority_order = self.actions.copy()
        # Rotate priority order every few steps
        priority_offset = (self.step_count // 5) % 3
        priority_order = priority_order[priority_offset:] + priority_order[:priority_offset]
        
        # Select action: use prioritized action if distance significant, else use largest distance
        selected_action = None
        for action in priority_order:
            distance = action_distances[action]
            if distance > 0.15:  # Threshold to consider this action
                selected_action = action
                break
        
        # If no action above threshold, select by largest distance
        if selected_action is None:
            selected_action = max(action_distances.items(), key=lambda x: x[1])[0]
        
        # Execute the selected action toward optimal
        if selected_action == 'adjust_temperature':
            adjustment = temp_step_size if temp_diff > 0 else -temp_step_size
            return {'action_type': 'adjust_temperature', 'value': float(adjustment)}
            
        elif selected_action == 'adjust_ph':
            adjustment = ph_step_size if ph_diff > 0 else -ph_step_size
            return {'action_type': 'adjust_ph', 'value': float(adjustment)}
            
        else:  # adjust_mutation
            adjustment = mutation_step_size if mutation_diff > 0 else -mutation_step_size
            return {'action_type': 'adjust_mutation', 'value': float(adjustment)}

def make_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make HTTP request to environment"""
    url = f"{ENV_URL}/{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request error: {e}")
        return {"error": str(e)}

def run_episode(task: str = "medium", seed: Optional[int] = None) -> Dict[str, float]:
    """Run a single episode with the optimization agent"""
    
    print(f"[START] Running episode with task={task}, seed={seed}")
    
    # Reset environment
    reset_data = {"task": task}
    if seed is not None:
        reset_data["seed"] = seed
    
    result = make_request("reset", "POST", reset_data)
    if "error" in result:
        print(f"Reset failed: {result['error']}")
        return {"reward": 0.0, "performance": 0.0, "success": False}
    
    state = result.get("state", {})
    print(f"Reset successful. Initial state: temp={state.get('temperature', 0):.1f}, "
          f"pH={state.get('ph', 0):.2f}, mutation={state.get('mutation_level', 0):.2f}")
    
    agent = ImprovedOptimizationAgent()
    total_reward = 0.0
    
    for step in range(MAX_STEPS):
        # Get action from agent
        action = agent.get_action(state)
        
        # Take step - wrap action in proper format
        step_payload = {"action": action}
        step_result = make_request("step", "POST", step_payload)
        if "error" in step_result:
            print(f"Step failed: {step_result['error']}")
            break
        
        # Update state and tracking
        state = step_result.get("state", {})
        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        total_reward += reward
        
        value_display = f"{action.get('value', 0.0):.2f}" if 'value' in action else "N/A"
        print(f"[STEP] action_type={action['action_type']} "
              f"value={value_display} "
              f"reward={reward:.3f} done={done}")
        
        if done:
            break
    
    # Final results
    final_performance = state.get("performance_score", 0.0)
    
    # Task-specific success thresholds
    task_thresholds = {
        "easy": 0.8,
        "medium": 0.75,
        "hard": 0.7
    }
    success_threshold = task_thresholds.get(task, 0.75)
    success = final_performance >= success_threshold
    
    print(f"[END] Episode completed. Total reward={total_reward:.3f}, "
          f"Final performance={final_performance:.3f}, Success={success}")
    
    return {
        "reward": total_reward,
        "performance": final_performance,
        "success": success
    }

def main():
    """Main execution function"""
    import sys
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Parse command line arguments
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    # Run episodes for all tasks
    tasks = ["easy", "medium", "hard"]
    all_results = []
    
    print("=" * 60)
    print(f"Starting OpenEnv Biological Optimization Inference")
    print(f"Improved Agent with LLM Support (when available)")
    print(f"Running {num_runs} seeds per task")
    print("=" * 60)
    
    episode_count = 0
    for task in tasks:
        task_results = []
        print(f"\n{'=' * 60}")
        print(f"Task: {task} (running {num_runs} seeds)")
        print("=" * 60)
        
        for seed_num in range(num_runs):
            episode_count += 1
            seed = 42 + seed_num  # Different seed for each run
            
            print(f"\n[{episode_count}] {task.upper()} Seed {seed_num+1}/{num_runs}")
            result = run_episode(task=task, seed=seed)
            all_results.append({
                "episode": episode_count,
                "task": task,
                "seed": seed,
                **result
            })
            task_results.append(result)
            time.sleep(0.5)
        
        # Task summary
        task_rewards = [r["reward"] for r in task_results]
        task_perfs = [r["performance"] for r in task_results]
        successes = sum(1 for r in task_results if r["success"])
        
        print(f"\n  Task {task} Summary:")
        print(f"    Avg Reward: {sum(task_rewards)/len(task_rewards):.3f}")
        print(f"    Avg Performance: {sum(task_perfs)/len(task_perfs):.3f}")
        print(f"    Success Rate: {successes}/{num_runs}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    results = all_results
    total_reward = 0.0
    successful_episodes = 0
    task_thresholds = {
        "easy": 0.8,
        "medium": 0.75,
        "hard": 0.7
    }
    
    task_stats = {task: {"rewards": [], "perfs": [], "successes": 0} for task in tasks}
    
    for result in results:
        episode_num = result["episode"]
        task = result["task"]
        reward = result["reward"]
        performance = result["performance"]
        success = result["success"]
        threshold = task_thresholds[task]
        
        total_reward += reward
        task_stats[task]["rewards"].append(reward)
        task_stats[task]["perfs"].append(performance)
        
        if success:
            successful_episodes += 1
            task_stats[task]["successes"] += 1
            
        status = "✓" if success else "✗"
        print(f"{status} Episode {episode_num:2d} ({task:6s}): reward={reward:6.3f}, "
              f"perf={performance:.3f} (target: {threshold:.2f})")
    
    print(f"\n{'=' * 60}")
    print("Task Statistics:")
    print("=" * 60)
    for task in tasks:
        stats = task_stats[task]
        if stats["rewards"]:
            avg_reward = sum(stats["rewards"]) / len(stats["rewards"])
            avg_perf = sum(stats["perfs"]) / len(stats["perfs"])
            success_rate = stats["successes"] / len(stats["rewards"])
            print(f"{task.upper():6s}: Avg Reward={avg_reward:.3f}, Avg Performance={avg_perf:.3f}, "
                  f"Success Rate={success_rate*100:.0f}% ({stats['successes']}/{len(stats['rewards'])})")
    
    print(f"\n{'=' * 60}")
    print("Overall Results:")
    print("=" * 60)
    print(f"- Total Episodes: {len(results)}")
    print(f"- Average Reward: {total_reward/len(results):.3f}")
    print(f"- Successful Episodes: {successful_episodes}/{len(results)}")
    print(f"- Success Rate: {successful_episodes/len(results)*100:.1f}%")
    print(f"- Final Score: {(total_reward/len(results))*100:.2f}")
    
    
    # Return final normalized score
    normalized_score = max(0.0, min(1.0, total_reward / len(results) / 10.0))
    print(f"- Normalized Score: {normalized_score:.3f}")
    
    return normalized_score

if __name__ == "__main__":
    try:
        score = main()
        print(f"\nFinal Score: {score:.3f}")
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Final Score: 0.000")
        
        