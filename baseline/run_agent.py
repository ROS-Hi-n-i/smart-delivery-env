from env.environment import DeliveryEnv

env = DeliveryEnv()
state = env.reset()

total_reward = 0
steps = 0

# simple heuristic agent
actions = ["move_A", "pick", "move_B", "pick", "move_C", "pick", "deliver"]

for action in actions:
    state, reward, done, _ = env.step(action)
    total_reward += reward
    steps += 1
    print(action, reward)

    if done:
        break

print("Total Reward:", total_reward)
print("Steps:", steps)
