def grade_easy(state):
    delivered = sum(p.delivered for p in state.packages)
    return delivered / len(state.packages)
