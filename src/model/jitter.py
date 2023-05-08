import random

def apply_jitter(state):
    state[0][0] = state.data[0][0] + ((random.random()-0.5) * 0.1)
    state.data[0][1] = state.data[0][1] + ((random.random()-0.5) * 0.002)
    return state