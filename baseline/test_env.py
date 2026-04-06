from env.environment import DeliveryEnv

env = DeliveryEnv()
state = env.reset()

print("Initial State:", state)

state, reward, done, _ = env.step("move_A")
print("After move:", state, reward)

state, reward, done, _ = env.step("pick")
print("After pick:", state, reward)

state, reward, done, _ = env.step("deliver")
print("After deliver:", state, reward, done)
