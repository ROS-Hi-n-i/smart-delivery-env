def grade_medium(state, steps):
    delivered = sum(p.delivered for p in state.packages)
    efficiency = max(0, 1 - steps / 20)
    return (delivered / len(state.packages)) * 0.7 + efficiency * 0.3
