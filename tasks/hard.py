def grade_hard(state, steps, total_reward):
    delivered = sum(p.delivered for p in state.packages)

    delivery_score = delivered / len(state.packages)
    efficiency_score = max(0, 1 - steps / 20)
    reward_score = max(0, total_reward / 5)

    return (delivery_score * 0.5 +
            efficiency_score * 0.3 +
            reward_score * 0.2)
