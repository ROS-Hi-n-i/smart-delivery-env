import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.environment import DeliveryEnv
from tasks.easy import grade_easy
from tasks.medium import grade_medium
from tasks.hard import grade_hard
# REQUIRED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "delivery-agent")

TASK_NAME = os.getenv("TASK", "easy")

ACTIONS = ["move_A", "move_B", "move_C", "pick", "deliver"]


# LOG FUNCTIONS 
def log_start():
    print(f"[START] task={TASK_NAME} env=smart-delivery model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def choose_action(state):
    pending = [p for p in state.packages if not p.delivered]

    if not pending:
        return "deliver"

    target = max(pending, key=lambda p: p.priority)

    if state.agent.location != target.location:
        return f"move_{target.location}"

    if target.id not in state.agent.carrying:
        return "pick"

    # prevent repeated delivery
    if any(p.id in state.agent.carrying for p in state.packages):
        return "deliver"

    return "move_A"
    
def explain_action(state, action):
    if action.startswith("move"):
        return "Moving towards target location"

    if action == "pick":
        return "Picking package at current location"

    if action == "deliver":
        for p in state.packages:
            if p.priority == 2 and not p.delivered:
                return "Delivering high priority package"
        return "Delivering normal package"

    return "Default action"


def run_episode():
    env = DeliveryEnv()
    state = env.reset()

    rewards = []
    steps = 0

    log_start()

    while True:
        steps += 1

        action = choose_action(state)
        state, reward, done, _ = env.step(action)

        rewards.append(reward)

        reason = explain_action(state, action)

        log_step(steps, action, reward, done)

        print(f"[DEBUG] reason={reason}", flush=True)

        if done or steps >= 20:
            break

    # SCORING 
    if TASK_NAME == "easy":
        score = grade_easy(state)
    elif TASK_NAME == "medium":
        score = grade_medium(state, steps)
    else:
        score = grade_hard(state, steps, sum(rewards))

    success = score > 0.3  # threshold

    log_end(success, steps, rewards)


if __name__ == "__main__":
    run_episode()
