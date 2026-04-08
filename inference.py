#!/usr/bin/env python3
"""
OpenEnv Inference Agent - Optimized for HuggingFace Space API
Robust to network failures and LLM unavailability.
"""

import os
import sys
import json
import requests
import time
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI


API_BASE_URL = "https://whyvek-bio-env.hf.space"
MAX_STEPS = 50
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LLM_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN", "test-token")

try:
    client = OpenAI(base_url=LLM_BASE_URL, api_key=HF_TOKEN)
except:
    client = None


class BiologicalOptimizationAgent:
    """Rule-based deterministic state machine for black-box optimization."""

    def __init__(self):
        # Best state memory
        self.best_reward      = -float('inf')
        self.best_action_type = "adjust_temperature"
        self.best_direction   = 1.0

        # Exploration state
        self.step_size    = 0.2
        self.direction    = 1.0
        self.action_types = ["adjust_temperature", "adjust_ph", "adjust_mutation"]
        self.action_index = 0

        # Peak-lock state
        self.peak_locked    = False
        self.peak_direction = 1.0   # independent direction used only inside peak-lock
        self.peak_steps     = 0     # steps spent in peak-lock

        # Drift detection
        self.drift_counter    = 0
        self.DRIFT_THRESHOLD  = 0.15   # drop fraction that starts counting drift
        self.DRIFT_PATIENCE   = 3      # steps before anti-drift fires
        self.PEAK_ESCAPE      = 6      # consecutive drift steps before exiting peak-lock

        # Previous step memory
        self.prev_reward      = None
        self.prev_action_type = "adjust_temperature"
        self.prev_direction   = 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_best(self, reward: float, action_type: str, direction: float):
        """Record best state seen so far. Called at top of get_action() with
        the reward that RESULTED from the previous action."""
        if reward > self.best_reward:
            self.best_reward      = reward
            self.best_action_type = action_type
            self.best_direction   = direction

    def _tick_drift(self, current_reward: float):
        """Increment or reset drift counter based on current reward vs best."""
        if self.prev_reward is not None and self.best_reward > 0:
            if current_reward < self.best_reward * (1.0 - self.DRIFT_THRESHOLD):
                self.drift_counter += 1
            else:
                self.drift_counter = 0

    # ------------------------------------------------------------------
    # Main action selection
    # ------------------------------------------------------------------

    def get_action(self, state: Dict[str, Any], rewards: List[float]) -> Dict[str, Any]:
        """
        Priority order (strictly if / elif):
          R1  Termination acceleration  — reward > 1.5
          R2  Peak-lock                 — reward >= 0.85 OR already locked
                                          (exits if drift_counter >= PEAK_ESCAPE)
          R3  Anti-drift                — drift_counter >= DRIFT_PATIENCE
          R4  Default exploration
        """
        current_reward = rewards[-1] if rewards else 0.0

        # ── Step A: update best with the reward that just arrived ────────
        if self.prev_reward is not None:
            self._update_best(current_reward,
                              self.prev_action_type,
                              self.prev_direction)

        # ── Step B: drift accounting (runs regardless of peak state) ─────
        self._tick_drift(current_reward)

        # ── Step C: select action ────────────────────────────────────────

        # R1 — Termination acceleration
        if current_reward > 1.5:
            action_type = self.best_action_type
            direction   = self.best_direction
            value       = direction * self.step_size

        # R2 — Peak-lock (with forced escape after PEAK_ESCAPE drift steps)
        elif (current_reward >= 0.85 or self.peak_locked) \
                and self.drift_counter < self.PEAK_ESCAPE:

            if not self.peak_locked:
                # First entry: adopt direction that produced the peak
                self.peak_locked    = True
                self.peak_direction = self.prev_direction
                self.peak_steps     = 0

            self.peak_steps += 1

            # Reverse peak_direction when reward declines inside peak region
            if self.prev_reward is not None and current_reward < self.prev_reward:
                self.peak_direction *= -1.0

            # Shrink step slowly to fine-tune around the peak
            self.step_size = max(0.01, self.step_size * 0.93)

            action_type = self.best_action_type
            direction   = self.peak_direction
            value       = direction * self.step_size

        # R3 — Anti-drift: return toward best region
        else:
            if self.peak_locked:
                # Forced escape from failed peak-lock
                self.peak_locked    = False
                self.peak_direction = 1.0
                self.peak_steps     = 0

            if self.drift_counter >= self.DRIFT_PATIENCE:
                action_type        = self.best_action_type
                direction          = self.best_direction * -1.0
                value              = direction * self.step_size
                self.drift_counter = 0

            # R4 — Default exploration
            else:
                if self.prev_reward is not None and current_reward < self.prev_reward:
                    self.direction *= -1.0
                    self.step_size  = max(0.01, self.step_size * 0.9)

                action_type = self.action_types[self.action_index % len(self.action_types)]
                direction   = self.direction
                value       = direction * self.step_size

        # ── Step D: persist state for next call ──────────────────────────
        self.prev_reward      = current_reward
        self.prev_action_type = action_type
        self.prev_direction   = direction
        self.action_index    += 1

        return {"action_type": action_type, "value": value}

    def get_llm_action(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Disabled — pure rule-based agent."""
        return None

    def get_fallback_action(self) -> Dict[str, Any]:
        return {"action_type": "adjust_temperature", "value": 0.2}


# ── Environment API helpers ────────────────────────────────────────────────

def reset_env(task: str = "medium", seed: int = 42,
              retries: int = 2) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    url = f"{API_BASE_URL}/reset"
    backoff_times = [1, 2, 4]
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json={"task": task, "seed": seed}, timeout=10)
            if resp.status_code == 200:
                return True, resp.json(), None
            if attempt < retries:
                time.sleep(backoff_times[attempt])
        except Exception:
            if attempt < retries:
                time.sleep(backoff_times[attempt])

    fallback_state = {
        "state": {
            "temperature": 25.0, "ph": 7.0, "mutation_level": 0.5,
            "performance_score": 0.0, "steps_count": 0, "stability_count": 0
        },
        "episode_info": {"task": task, "max_steps": 50}
    }
    return False, fallback_state, "reset_fallback"


def step_env(action: Dict[str, Any],
             retries: int = 2) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    url = f"{API_BASE_URL}/step"
    backoff_times = [1, 2, 4]
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json={"action": action}, timeout=10)
            if resp.status_code == 200:
                return True, resp.json(), None
            if attempt < retries:
                time.sleep(backoff_times[attempt])
        except Exception:
            if attempt < retries:
                time.sleep(backoff_times[attempt])
    return False, {}, "timeout"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    rewards: List[str]  = []
    reward_floats: List[float] = []
    steps_taken = 0
    state       = None

    try:
        print(f"[START] task=medium env={API_BASE_URL} model={MODEL_NAME}")

        ok, data, _ = reset_env("medium", 42, retries=2)
        state       = data.get("state", {})

        agent        = BiologicalOptimizationAgent()
        loop_executed = False

        for step_num in range(1, MAX_STEPS + 1):
            loop_executed = True
            steps_taken   = step_num

            try:
                llm_action = agent.get_llm_action(state)
                action = (llm_action
                          if llm_action
                             and 'action_type' in llm_action
                             and 'value' in llm_action
                          else agent.get_action(state, reward_floats))
            except Exception:
                action = agent.get_fallback_action()

            try:
                ok, step_data, err = step_env(action, retries=2)
                if ok:
                    state     = step_data.get("state", {})
                    reward    = float(step_data.get("reward", 0.0))
                    done      = step_data.get("done", False)
                    error_str = "null"
                else:
                    reward    = 0.0
                    done      = False
                    error_str = f'"{err}"' if err else '"timeout"'
            except Exception:
                reward    = 0.0
                done      = False
                error_str = '"exception"'

            reward_floats.append(reward)
            rewards.append(f"{reward:.2f}")
            done_str = "true" if done else "false"
            print(f"[STEP] step={step_num} "
                  f"action={action.get('action_type', 'unknown')} "
                  f"reward={reward:.2f} done={done_str} error={error_str}")

            if done:
                break

        if not loop_executed:
            steps_taken = 1
            try:
                action    = {"action_type": "adjust_temperature", "value": 0.5}
                ok, step_data, err = step_env(action, retries=2)
                reward    = float(step_data.get("reward", 0.0)) if ok else 0.0
                error_str = "null" if ok else f'"{err}"'
                state     = step_data.get("state", {}) if ok else state
            except Exception:
                reward    = 0.0
                error_str = '"exception"'
            rewards.append(f"{reward:.2f}")
            print(f"[STEP] step=1 action=adjust_temperature "
                  f"reward={reward:.2f} done=false error={error_str}")

        perf        = state.get("performance_score", 0.0) if state else 0.0
        success_str = "true" if perf >= 0.8 else "false"
        print(f"[END] success={success_str} steps={steps_taken} "
              f"rewards={','.join(rewards)}")

    except Exception:
        print(f"[END] success=false steps={steps_taken} "
              f"rewards={','.join(rewards) if rewards else ''}")


if __name__ == "__main__":
    main()