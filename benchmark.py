#!/usr/bin/env python3
"""Benchmarking script for BiologicalOptimizationEnv using rule-based agent"""

import requests
import statistics
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import argparse


@dataclass
class EpisodeResult:
    """Result from a single episode"""
    task: str
    seed: int
    success: bool
    performance: float
    steps: int
    total_reward: float
    rewards: List[float]


class RuleBasedAgent:
    """Simple rule-based agent that moves toward optimal values"""
    
    OPTIMAL_TEMP = 37.0
    OPTIMAL_PH = 7.4
    OPTIMAL_MUTATION = 0.3
    
    def get_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get action using simple rules"""
        temp = state.get('temperature', 25.0)
        ph = state.get('ph', 7.0)
        mutation = state.get('mutation_level', 0.5)
        performance = state.get('performance_score', 0.0)
        stability = state.get('stability_count', 0)
        
        # Run experiment when conditions are good
        if performance >= 0.85 and stability >= 2:
            return {'action_type': 'run_experiment', 'value': 0}
        
        # Calculate deltas to optimal
        temp_diff = self.OPTIMAL_TEMP - temp
        ph_diff = self.OPTIMAL_PH - ph
        mutation_diff = self.OPTIMAL_MUTATION - mutation
        
        # Prioritize the largest delta
        abs_diffs = [
            (abs(temp_diff), 'temperature', temp_diff),
            (abs(ph_diff), 'ph', ph_diff),
            (abs(mutation_diff), 'mutation', mutation_diff)
        ]
        abs_diffs.sort(reverse=True)
        
        _, param, diff = abs_diffs[0]
        
        if param == 'temperature':
            value = max(-5.0, min(5.0, diff))
            return {'action_type': 'adjust_temperature', 'value': value}
        elif param == 'ph':
            value = max(-1.0, min(1.0, diff))
            return {'action_type': 'adjust_ph', 'value': value}
        else:  # mutation
            value = max(-0.2, min(0.2, diff))
            return {'action_type': 'adjust_mutation', 'value': value}


class EnvironmentClient:
    """Client for interacting with the environment API"""
    
    def __init__(self, env_url: str):
        self.env_url = env_url.rstrip('/')
        
    def reset(self, task: str = "medium") -> Dict[str, Any]:
        """Reset environment with task"""
        try:
            response = requests.post(f"{self.env_url}/reset", json={"task": task}, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Reset failed: {response.status_code}")
            return response.json()
        except Exception as e:
            raise Exception(f"Reset error: {e}")
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Take environment step"""
        payload = {
            "action": {
                "action_type": action.get('action_type', ''),
                "value": action.get('value', 0)
            }
        }
        
        try:
            response = requests.post(f"{self.env_url}/step", json=payload, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Step failed: {response.status_code}")
            return response.json()
        except Exception as e:
            raise Exception(f"Step error: {e}")


def run_episode(env_client: EnvironmentClient, agent: RuleBasedAgent, 
                task: str, seed: int, max_steps: int = 50) -> EpisodeResult:
    """Run a single episode"""
    
    try:
        reset_result = env_client.reset(task=task)
        state = reset_result.get("state", {})
    except Exception as e:
        return EpisodeResult(
            task=task, seed=seed, success=False, performance=0.0,
            steps=0, total_reward=-1.0, rewards=[]
        )
    
    rewards = []
    total_reward = 0.0
    final_success = False
    
    for step_num in range(max_steps):
        try:
            action = agent.get_action(state)
            step_result = env_client.step(action)
            
            state = step_result.get("state", {})
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            info = step_result.get("info", {})
            
            rewards.append(reward)
            total_reward += reward
            
            # Track actual success from environment
            final_success = info.get("success", False)
            
            if done:
                break
                
        except Exception as e:
            break
    
    final_performance = state.get("performance_score", 0.0)
    
    return EpisodeResult(
        task=task,
        seed=seed,
        success=final_success,  # Use actual environment success flag from info
        performance=final_performance,
        steps=len(rewards),
        total_reward=total_reward,
        rewards=rewards
    )


def benchmark(env_url: str, episodes_per_task: int = 5, tasks: List[str] = None) -> None:
    """Run benchmark across tasks"""
    
    if tasks is None:
        tasks = ["easy", "medium", "hard"]
    
    print("\n" + "=" * 80)
    print("OPENENV BIOLOGICAL OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print(f"Environment: {env_url}")
    print(f"Episodes per task: {episodes_per_task}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80 + "\n")
    
    agent = RuleBasedAgent()
    env_client = EnvironmentClient(env_url)
    
    # Verify server is running
    try:
        requests.get(f"{env_url}/", timeout=2)
    except:
        print("❌ ERROR: Server not responding at", env_url)
        return
    
    all_results = []
    
    for task in tasks:
        print(f"[{task.upper()}] Running {episodes_per_task} episodes...")
        task_results = []
        
        for seed_offset in range(episodes_per_task):
            seed = hash(f"{task}_{seed_offset}") % 10000
            result = run_episode(env_client, agent, task, seed, max_steps=50)
            task_results.append(result)
            all_results.append(result)
            
            status = "✓" if result.success else "✗"
            print(f"  {status} Episode {seed_offset+1}: steps={result.steps:2d}, "
                  f"reward={result.total_reward:7.3f}, perf={result.performance:.3f}")
        
        # Task summary
        successes = sum(1 for r in task_results if r.success)
        avg_steps = statistics.mean(r.steps for r in task_results)
        avg_reward = statistics.mean(r.total_reward for r in task_results)
        avg_perf = statistics.mean(r.performance for r in task_results)
        
        print(f"  Task Summary ({task}):")
        print(f"    Success Rate: {successes}/{episodes_per_task}")
        print(f"    Avg Steps: {avg_steps:.1f}")
        print(f"    Avg Reward: {avg_reward:.3f}")
        print(f"    Avg Performance: {avg_perf:.3f}\n")
    
    # Overall summary
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    total_episodes = len(all_results)
    successes = sum(1 for r in all_results if r.success)
    avg_steps = statistics.mean(r.steps for r in all_results)
    avg_reward = statistics.mean(r.total_reward for r in all_results)
    avg_perf = statistics.mean(r.performance for r in all_results)
    
    # Task differentiation
    task_avg_rewards = {}
    for task in tasks:
        task_rewards = [r.total_reward for r in all_results if r.task == task]
        if task_rewards:
            task_avg_rewards[task] = statistics.mean(task_rewards)
    
    discriminative = len(set(f"{v:.1f}" for v in task_avg_rewards.values())) > 1
    
    print(f"Total Episodes: {total_episodes}")
    print(f"Success Rate: {successes}/{total_episodes} ({100*successes/total_episodes:.1f}%)")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Average Performance: {avg_perf:.3f}")
    
    print(f"\nTask Differentiation:")
    for task in sorted(task_avg_rewards.keys()):
        print(f"  {task.upper():6s}: {task_avg_rewards[task]:.3f} avg reward")
    
    print(f"\nEnvironment Quality:")
    print(f"  Discriminative? {'✓' if discriminative else '✗'}")
    print(f"  Episode length: {min(r.steps for r in all_results)}-{max(r.steps for r in all_results)} steps")
    
    if avg_steps > 10 and successes >= total_episodes * 0.5 and discriminative:
        print(f"\n✓ ENVIRONMENT QUALITY: GOOD")
    elif avg_steps > 5:
        print(f"\n⚠️  ENVIRONMENT QUALITY: ACCEPTABLE")
    else:
        print(f"\n❌ ENVIRONMENT QUALITY: NEEDS IMPROVEMENT")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenEnv environment")
    parser.add_argument("--env-url", default="http://127.0.0.1:7860",
                       help="Environment URL (default: http://127.0.0.1:7860)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Episodes per task (default: 5)")
    parser.add_argument("--tasks", default="easy,medium,hard",
                       help="Tasks to benchmark (default: easy,medium,hard)")
    
    args = parser.parse_args()
    
    tasks = [t.strip() for t in args.tasks.split(",")]
    benchmark(args.env_url, episodes_per_task=args.episodes, tasks=tasks)


if __name__ == "__main__":
    main()
