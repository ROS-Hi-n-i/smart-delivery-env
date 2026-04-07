import os
import sys
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.environment import DeliveryEnv
from tasks.easy import grade_easy
from tasks.medium import grade_medium
from tasks.hard import grade_hard

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

TASK_NAME = os.getenv("TASK", "easy")
ACTIONS = ["move_A", "move_B", "move_C", "pick", "deliver"]


def log_start():
    print(f"[START] task={TASK_NAME} env=smart-delivery model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error=None):
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}",
        flush=True,
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def choose_action(state):
    pending = [p for p in state.packages if not p.delivered]

    if not pending:
        return "deliver"

    target = max(pending, key=lambda p: p.priority)

    if state.agent.location != target.location:
        return f"move_{target.location}"

    if target.id not in state.agent.carrying:
        return "pick"

    if any(p.id in state.agent.carrying for p in state.packages):
        return "deliver"

    return "move_A"


def run_episode():
    env = DeliveryEnv()
    rewards = []
    steps = 0
    success = False

    try:
        state = env.reset()
        log_start()

        while True:
            action = choose_action(state)

            try:
                state, reward, done, _ = env.step(action)
                error = None
            except Exception as e:
                reward = 0.00
                done = False
                error = str(e)
                log_step(steps + 1, action, reward, done, error=error)
                break

            steps += 1
            rewards.append(reward)
            log_step(steps, action, reward, done, error=None)

            if done or steps >= 20:
                break

        if TASK_NAME == "easy":
            score = grade_easy(state)
        elif TASK_NAME == "medium":
            score = grade_medium(state, steps)
        else:
            score = grade_hard(state, steps, sum(rewards))

        success = score > 0.3

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success, steps, rewards)


if __name__ == "__main__":
    run_episode()
