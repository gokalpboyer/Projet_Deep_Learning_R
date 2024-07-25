import numpy as np


def sarsa(env, gamma=0.99, alpha=0.1, epsilon=0.1, episodes=1000):
    Q = np.zeros((env.length, 2))
    policy = np.zeros(env.length, dtype=int)

    def choose_action(state):
        if np.random.rand() < epsilon:
            return np.random.choice([0, 1])
        else:
            return policy[state]

    for _ in range(episodes):
        state = env.reset()
        action = choose_action(state)
        while True:
            next_state, reward, done = env.step(action)
            next_action = choose_action(next_state)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            policy[state] = np.argmax(Q[state])
            state, action = next_state, next_action
            if done:
                break
    return policy, Q
