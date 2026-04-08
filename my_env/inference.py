"""
Inference script for adaptive experimental design OpenEnv environment.
Calls the environment over HTTP. Do not import server-side classes here.
Usage:
export HF_TOKEN=your_token
export MY_ENV_URL=http://localhost:8000
export MY_ENV_TASK=medium
python inference.py
"""
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional
import httpx
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_TASK") or "medium"
BENCHMARK = "adaptive_experimental_design"
MAX_STEPS = int(os.getenv("MAX_STEPS") or "30")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE") or "0.2")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS") or "120")
SEED = int(os.getenv("SEED") or "7")
ENV_URL = os.getenv("MY_ENV_URL") or "http://localhost:8000"

ACTION_LABELS = {
    0: "0:increase_temperature",
    1: "1:decrease_temperature",
    2: "2:increase_pH",
    3: "3:decrease_pH",
    4: "4:mutate_sequence",
    5: "5:run_experiment",
}

SYSTEM_PROMPT = textwrap.dedent("""
You are controlling an adaptive experimental design environment.
Choose exactly one action_id per step:
0 increase temperature (optimal: 42.0)
1 decrease temperature
2 increase pH (optimal: 6.5)
3 decrease pH
4 mutate sequence (optimal level: 3)
5 run experiment (ends episode — only use when task_score is high enough)
Output only JSON: {"action_id": <0-5>, "reason": "<brief>"}.
Maximize task_score before running the experiment.
Do not repeat the same action more than twice in a row.
""").strip()

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={clamp01(score):.3f} rewards={rewards_str}",
        flush=True,
    )

def env_reset() -> Dict[str, Any]:
    r = httpx.post(
        f"{ENV_URL}/reset",
        json={"task_name": TASK_NAME, "seed": SEED},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()

def env_step(action_id: int) -> Dict[str, Any]:
    r = httpx.post(
        f"{ENV_URL}/step",
        json={"action": {"action_id": action_id}},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()

def parse_action_id(raw: str) -> Optional[int]:
    text = (raw or "").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            v = payload.get("action_id")
            if isinstance(v, int) and 0 <= v <= 5:
                return v
    except Exception:
        pass
    m = re.search(r"\b([0-5])\b", text)
    return int(m.group(1)) if m else None

def run_allowed(task: str, step: int, score: float, used: set) -> bool:
    t = task.lower()
    if t == "easy":
        return step >= 2 and score >= 0.95
    if t == "medium":
        return step >= 6 and len(used) >= 3 and score >= 0.88
    return step >= 20 and {0, 3, 4}.issubset(used) and score >= 0.93

def heuristic(obs: Dict, step: int, used: set) -> int:
    temp = float(obs.get("temperature", 42.0))
    ph = float(obs.get("pH", 6.5))
    mut = int(obs.get("mutation_level", 3))
    score = clamp01(float(obs.get("task_score", 0.0)))
    task = str(obs.get("task_name", TASK_NAME)).lower()
    
    # For medium/hard tasks, prioritize using 3 distinct actions early
    if task == "medium" and step <= 6 and len(used) < 3:
        for c in (2, 3, 4, 0, 1):
            if c not in used:
                return c
    
    # Check if we can run the experiment
    if run_allowed(task, step, score, used):
        return 5
    
    # Prioritize reaching optimal temperature (42)
    temp_error = abs(42.0 - temp)
    if temp_error >= 1.0:
        return 0 if temp < 42.0 else 1
    
    # Then prioritize pH (6.5)
    ph_error = abs(6.5 - ph)
    if ph_error >= 0.1:
        return 2 if ph < 6.5 else 3
    
    # Then mutation level (3)
    mut_error = abs(3 - mut)
    if mut_error > 0:
        fwd = (3 - mut) % 7
        return 4 if fwd != 0 else 0
    
    # If very close to optimal, try to run experiment
    if temp_error < 0.5 and ph_error < 0.05 and mut_error == 0:
        return 5
    
    # Otherwise explore unused actions
    if task in ("medium", "hard"):
        for c in (0, 3, 4, 2, 1):
            if c not in used:
                return c
    return 0

def get_llm_action(client: Any, obs: Dict, step: int, used: set, history: List[str]) -> Optional[int]:
    if client is None:
        return None
    task = str(obs.get("task_name", TASK_NAME))
    score = clamp01(float(obs.get("task_score", 0.0)))
    prompt = (
        f"task={task} step={step}/{MAX_STEPS}\n"
        f"temperature={obs.get('temperature'):.2f} pH={obs.get('pH'):.2f} "
        f"mutation_level={obs.get('mutation_level')} score={score:.4f} "
        f"steps_remaining={obs.get('steps_remaining')}\n"
        f"used_actions={sorted(used)}\n"
        f"success_condition={obs.get('success_condition', '')}\n"
        f"recent:\n" + "\n".join(history[-4:])
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return parse_action_id(resp.choices[0].message.content or "")
    except Exception:
        return None

def choose(client: Any, obs: Dict, step: int, used: set, history: List[str]) -> int:
    h = heuristic(obs, step, used)
    task = str(obs.get("task_name", TASK_NAME)).lower()
    score = clamp01(float(obs.get("task_score", 0.0)))
    llm = get_llm_action(client, obs, step, used, history)
    if llm is None:
        return h
    if llm == 5 and not run_allowed(task, step, score, used):
        return h
    return llm

def main() -> None:
    rewards: List[float] = []
    history: List[str] = []
    used: set = set()
    steps_taken = 0
    best_score = 0.0
    success = False
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    client = None
    if OpenAI is not None and API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception:
            pass

    obs: Dict[str, Any] = {}
    try:
        resp = env_reset()
        obs = resp.get("observation", {}) if "observation" in resp else resp

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action_id = choose(client, obs, step, used, history)
            action_str = ACTION_LABELS.get(action_id, f"{action_id}:unknown")

            reward = 0.0
            done = True
            error: Optional[str] = None

            try:
                result = env_step(action_id)
                # Unwrap observation from server response
                obs_data = result.get("observation", {}) if "observation" in result else result
                reward = max(-1.0, min(1.0, float(result.get("reward", 0.0))))
                done = bool(result.get("done", False))
                # Merge reward and done into observation for logging
                obs = {**obs_data, "reward": reward, "done": done}
                used.add(action_id)
                score_now = clamp01(float(obs.get("task_score", 0.0)))
                best_score = max(best_score, score_now)
                
                # Calculate distance to optimal for debugging
                temp = float(obs.get("temperature", 42.0))
                ph = float(obs.get("pH", 6.5))
                mut = int(obs.get("mutation_level", 3))
                dist = abs(temp - 42.0) + abs(ph - 6.5) * 10.0 + abs(mut - 3)
                print(f"[STATE] T={temp:.1f} pH={ph:.2f} M={mut} score={score_now:.4f} dist={dist:.2f}", file=sys.stderr)
                history.append(
                    f"step={step} action={action_str} "
                    f"score={score_now:.4f} reward={reward:.2f}"
                )
            except Exception as exc:
                error = str(exc)
                done = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        success = bool(obs.get("success", False))

    except Exception as e:
        print(f"Episode error: {e}", file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, score=best_score, rewards=rewards)

if __name__ == "__main__":
    main()
