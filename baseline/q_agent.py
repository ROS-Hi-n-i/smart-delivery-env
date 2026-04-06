from env.environment import DeliveryEnv
import random

# possible actions
ACTIONS = ["move_A", "move_B", "move_C", "pick", "deliver"]

# Q-table (memory)
q_table = {}

# learning parameters
alpha = 0.1   # learning rate
gamma = 0.9   # future reward importance
epsilon = 0.2 # exploration

def get_state_key(state):
    return (state.agent.location, tuple(p.delivered for p in state.packages))

def choose_action(state_key):
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    return max(ACTIONS, key=lambda a: q_table.get((state_key, a), 0))

def update_q(state_key, action, reward, next_state_key):
    old_q = q_table.get((state_key, action), 0)
    future_q = max([q_table.get((next_state_key, a), 0) for a in ACTIONS])

    new_q = old_q + alpha * (reward + gamma * future_q - old_q)
    q_table[(state_key, action)] = new_q


# TRAINING LOOP
env = DeliveryEnv()

episodes = 200

for episode in range(episodes):
    state = env.reset()
    state_key = get_state_key(state)

    total_reward = 0

    while True:
        action = choose_action(state_key)
        next_state, reward, done, _ = env.step(action)

        next_state_key = get_state_key(next_state)

        update_q(state_key, action, reward, next_state_key)

        state_key = next_state_key
        total_reward += reward

        if done:
            break

    print(f"Episode {episode} → Reward: {total_reward}")

print("Training complete")

print("\n--- FINAL TEST ---")

state = env.reset()
state_key = get_state_key(state)

total_reward = 0

while True:
    action = max(ACTIONS, key=lambda a: q_table.get((state_key, a), 0))
    next_state, reward, done, _ = env.step(action)

    state_key = get_state_key(next_state)
    total_reward += reward

    if done:
        break

print("Final Score:", total_reward)
