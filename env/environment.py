from env.models import *
import random

class DeliveryEnv:
    def __init__(self):
        self.max_steps = 20
        self.reset()

    def reset(self):
        self.steps = 0
        self.state_data = EnvironmentState(
            agent=AgentState(location="warehouse", carrying=[]),
            packages=[
                Package(id=1, location="A", delivered=False, priority=2), # HIGH
                Package(id=2, location="B", delivered=False, priority=1),
                Package(id=3, location="C", delivered=False, priority=1)
            ]
        )
        return self.state()

    def state(self):
        return self.state_data

    def step(self, action):
        self.steps += 1
        reward = -0.05   # time penalty every step

        # Random traffic delay
        if random.random() < 0.1:
            return self.state(), reward - 0.2, False, {"event": "traffic"}

        # Movement
        if action.startswith("move_"):
            location = action.split("_")[1]
            self.state_data.agent.location = location
            reward -= 0.1   # fuel cost

        # Pick package
        elif action == "pick":
            for p in self.state_data.packages:
                if p.location == self.state_data.agent.location and not p.delivered:
                    if p.id not in self.state_data.agent.carrying:
                        self.state_data.agent.carrying.append(p.id)
                        reward += 0.3

        # Deliver package
        elif action == "deliver":
            for p in self.state_data.packages:
                if p.id in self.state_data.agent.carrying:
                    p.delivered = True

                    if p.priority == 2:
                        reward += 2.0   # HIGH priority reward 
                    else:
                        reward += 1.0  

        done = all(p.delivered for p in self.state_data.packages)

        # Late penalty
        if self.steps >= self.max_steps:
            done = True
            reward -= 1.0

        return self.state(), reward, done, {}
