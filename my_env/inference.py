"""LLM-driven inference script for adaptive experimental design."""

import json
import os
import re
import sys
import textwrap
from typing import Any, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
for path in (PROJECT_DIR, WORKSPACE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from server.my_env_environment import MyEnvironment
from models import MyAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_TASK") or os.getenv("MY_ENV_V4_TASK") or "medium"
BENCHMARK = os.getenv("MY_ENV_BENCHMARK") or "adaptive_experimental_design"
MAX_STEPS = int(os.getenv("MAX_STEPS") or "30")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE") or "0.2")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS") or "120")
SEED = int(os.getenv("SEED") or "7")

ACTION_LABELS = {
    0: "0:increase_temperature",
    1: "1:decrease_temperature",
    2: "2:increase_pH",
    3: "3:decrease_pH",
    4: "4:mutate_sequence",
    5: "5:run_experiment",
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling an adaptive experimental design environment.
    Choose exactly one action_id per step:
    0 increase temperature
    1 decrease temperature
    2 increase pH
    3 decrease pH
    4 mutate sequence
    5 run experiment (terminal)

    Output only JSON: {"action_id": <0-5>, "reason": "<brief>"}.
    Prefer actions that increase task_score and satisfy the task success condition.
    Avoid repeating the same action unless it clearly improves score.
    """
).strip()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={clamp01(score):.3f} rewards={rewards_str}",
        flush=True,
    )


def describe_state_change(prev_obs: Any, curr_obs: Any) -> str:
    if prev_obs is None or curr_obs is None:
        return "initial"

    d_temp = as_float(getattr(curr_obs, "temperature", 0.0)) - as_float(
        getattr(prev_obs, "temperature", 0.0)
    )
    d_ph = as_float(getattr(curr_obs, "pH", 0.0)) - as_float(getattr(prev_obs, "pH", 0.0))
    d_mut = as_int(getattr(curr_obs, "mutation_level", 0)) - as_int(
        getattr(prev_obs, "mutation_level", 0)
    )
    d_score = as_float(getattr(curr_obs, "task_score", 0.0)) - as_float(
        getattr(prev_obs, "task_score", 0.0)
    )
    return f"dT={d_temp:+.1f},dpH={d_ph:+.2f},dM={d_mut:+d},dS={d_score:+.4f}"


def parse_action_id(raw_text: str) -> Optional[int]:
    text = (raw_text or "").strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            value = payload.get("action_id")
            if isinstance(value, int) and 0 <= value <= 5:
                return value
    except Exception:
        pass

    match = re.search(r"\b([0-5])\b", text)
    if match:
        return int(match.group(1))
    return None


def run_allowed(task_name: str, step: int, score: float, used_actions: set[int], max_steps: int) -> bool:
    if step >= max_steps:
        return True

    normalized = (task_name or "medium").strip().lower()
    if normalized == "easy":
        return step >= 2 and score >= 0.95

    if normalized == "medium":
        return step >= 6 and len(used_actions) >= 3 and score >= 0.88

    required = {0, 3, 4}
    return step >= 20 and required.issubset(used_actions) and score >= 0.93


def heuristic_action(obs: Any, step: int, used_actions: set[int], max_steps: int) -> int:
    task_name = str(getattr(obs, "task_name", TASK_NAME)).lower()
    temperature = as_float(getattr(obs, "temperature", 42.0))
    ph = as_float(getattr(obs, "pH", 6.5))
    mutation = as_int(getattr(obs, "mutation_level", 3))
    score = clamp01(as_float(getattr(obs, "task_score", 0.0)))

    if run_allowed(task_name, step, score, used_actions, max_steps):
        return 5

    temp_error = 42.0 - temperature
    if abs(temp_error) >= 0.5:
        return 0 if temp_error > 0 else 1

    ph_error = 6.5 - ph
    if abs(ph_error) >= 0.05:
        return 2 if ph_error > 0 else 3

    forward_mut_steps = (3 - mutation) % 7
    if forward_mut_steps != 0:
        return 4

    if task_name == "medium" and step < 6:
        for candidate in (0, 3, 4, 2, 1):
            if candidate not in used_actions:
                return candidate

    if task_name == "hard" and step < 20:
        for candidate in (0, 3, 4):
            if candidate not in used_actions:
                return candidate
        return 0

    return 5 if step >= max_steps else 0


def build_user_prompt(obs: Any, step: int, used_actions: set[int], history: List[str], max_steps: int) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    task_name = str(getattr(obs, "task_name", TASK_NAME))
    success_condition = str(getattr(obs, "success_condition", ""))
    temperature = as_float(getattr(obs, "temperature", 0.0))
    ph = as_float(getattr(obs, "pH", 0.0))
    mutation = as_int(getattr(obs, "mutation_level", 0))
    score = clamp01(as_float(getattr(obs, "task_score", 0.0)))
    steps_remaining = as_int(getattr(obs, "steps_remaining", 0))

    return textwrap.dedent(
        f"""
        task={task_name}
        success_condition={success_condition}
        step={step}/{max_steps}
        state: temperature={temperature:.2f}, pH={ph:.2f}, mutation_level={mutation}, score={score:.4f}, steps_remaining={steps_remaining}
        used_actions={sorted(used_actions)}
        recent_transitions:
        {history_block}

        Select the single best next action_id.
        """
    ).strip()


def get_llm_action(client: Any, obs: Any, step: int, used_actions: set[int], history: List[str], max_steps: int) -> Optional[int]:
    if client is None:
        return None

    prompt = build_user_prompt(obs=obs, step=step, used_actions=used_actions, history=history, max_steps=max_steps)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return parse_action_id(raw)
    except Exception:
        return None


def choose_action(client: Any, obs: Any, step: int, used_actions: set[int], history: List[str], max_steps: int) -> int:
    heuristic = heuristic_action(obs=obs, step=step, used_actions=used_actions, max_steps=max_steps)
    llm_action = get_llm_action(
        client=client,
        obs=obs,
        step=step,
        used_actions=used_actions,
        history=history,
        max_steps=max_steps,
    )
    if llm_action is None:
        return heuristic

    task_name = str(getattr(obs, "task_name", TASK_NAME)).lower()
    score = clamp01(as_float(getattr(obs, "task_score", 0.0)))
    if llm_action == 5 and not run_allowed(task_name, step, score, used_actions, max_steps):
        return heuristic
    return llm_action


def main() -> None:
    rewards: List[float] = []
    transitions: List[str] = []
    used_actions: set[int] = set()
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    env = None
    obs = None
    client = None
    if OpenAI is not None and API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception:
            client = None

    try:
        env = MyEnvironment()
        obs = env.reset(seed=SEED, task_name=TASK_NAME)

        for step in range(1, MAX_STEPS + 1):
            if bool(obs.done):
                break

            action_id = choose_action(
                client=client,
                obs=obs,
                step=step,
                used_actions=used_actions,
                history=transitions,
                max_steps=MAX_STEPS,
            )
            action_str = ACTION_LABELS.get(action_id, f"{action_id}:unknown")

            reward = 0.0
            done = True
            step_error: Optional[str] = None
            prev_obs = obs

            try:
                obs = env.step(MyAction(action_id=action_id))
                reward = as_float(obs.reward, 0.0)
                done = bool(obs.done)
                used_actions.add(action_id)
                transitions.append(
                    (
                        f"step={step} action={action_str} "
                        f"change={describe_state_change(prev_obs, obs)} "
                        f"reward={reward:.2f} score={clamp01(as_float(getattr(obs, 'task_score', 0.0))):.3f}"
                    )
                )
            except Exception as exc:
                step_error = str(exc)
                done = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=step_error)

            if done:
                break

        final_score = clamp01(as_float(getattr(obs, "task_score", 0.0), 0.0)) if obs is not None else 0.0
        score = final_score
        success = bool(getattr(obs, "success", False)) if obs is not None else False

    except Exception as e:
        success = False
        score = 0.0
        print(f"Error: {e}")

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
