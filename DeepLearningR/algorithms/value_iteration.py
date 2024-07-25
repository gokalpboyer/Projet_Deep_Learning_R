import numpy as np


def value_iteration(env, gamma=0.99, theta=1e-6):
    value_table = np.zeros(env.length)
    policy = np.zeros(env.length, dtype=int)

    while True:
        delta = 0
        for state in range(env.length):
            v = value_table[state]
            value_list = []
            for action in [0, 1]:  # Actions: left, right
                env.state = state  # Revenir à l'état actuel
                next_state, reward, done = env.step(action)
                value = reward + gamma * value_table[next_state]
                value_list.append(value)
            value_table[state] = max(value_list)
            policy[state] = np.argmax(value_list)
            delta = max(delta, abs(v - value_table[state]))
        if delta < theta:
            break

    return policy, value_table
