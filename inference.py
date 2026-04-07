import os
from openai import OpenAI
from env.environment import DeliveryEnv
from tasks.easy import grade_easy
from tasks.medium import grade_medium
from tasks.hard import grade_hard

# ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

TASK_NAME = os.getenv("TASK", "easy")

def log_start():
    print(f"[START] task={TASK_NAME} env=smart-delivery model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# 🔥 LLM-based action (THIS WAS MISSING)
def get_action_from_llm(state):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a delivery agent."},
                {"role": "user", "content": f"State: {state}. Choose next action from [move_A, move_B, move_C, pick, deliver]"}
            ],
            max_tokens=10
        )
        action = response.choices[0].message.content.strip()

        # fallback safety
        if action not in ["move_A", "move_B", "move_C", "pick", "deliver"]:
            return "move_A"

        return action

    except Exception:
        return "move_A"


def run_episode():
    env = None
    rewards = []
    steps = 0
    success = False

    try:
        env = DeliveryEnv()
        state = env.reset()

        log_start()

        while True:
            steps += 1

            action = get_action_from_llm(state)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)

            log_step(steps, action, reward, done)

            if done or steps >= 20:
                break

        # SCORING
        if TASK_NAME == "easy":
            score = grade_easy(state)
        elif TASK_NAME == "medium":
            score = grade_medium(state, steps)
        else:
            score = grade_hard(state, steps, sum(rewards))

        success = score > 0.3

    finally:
        try:
            if env is not None:
                env.close()
        except:
            pass

        log_end(success, steps, rewards)


if __name__ == "__main__":
    run_episode()
