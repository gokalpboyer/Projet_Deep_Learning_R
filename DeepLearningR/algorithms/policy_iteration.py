import numpy as np

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    value_table = np.zeros(env.length)
    while True:
        delta = 0
        for state in range(env.length):
            v = value_table[state]
            action = policy[state]
            env.state = state
            next_state, reward, _ = env.step(action)
            value_table[state] = reward + gamma * value_table[next_state]
            delta = max(delta, abs(v - value_table[state]))
        if delta < theta:
            break
    return value_table

def policy_improvement(env, value_table, gamma=0.99):
    policy = np.zeros(env.length, dtype=int)
    for state in range(env.length):
        action_values = []
        for action in [0, 1]:
            env.state = state
            next_state, reward, _ = env.step(action)
            action_values.append(reward + gamma * value_table[next_state])
        policy[state] = np.argmax(action_values)
    return policy

def policy_iteration(env, gamma=0.99, theta=1e-6):
    policy = np.zeros(env.length, dtype=int)
    while True:
        value_table = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, value_table, gamma)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, value_table
