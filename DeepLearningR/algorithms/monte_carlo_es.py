import numpy as np
from collections import defaultdict

def generate_episode(env, policy):
    episode = []
    state = env.reset()
    for _ in range(100):
        action = policy[state]
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def monte_carlo_es(env, gamma=0.99, episodes=1000):
    Q = defaultdict(lambda: np.zeros(env.length))
    policy = np.zeros(env.length, dtype=int)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for _ in range(episodes):
        episode = generate_episode(env, policy)
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if not (state, action) in [(x[0], x[1]) for x in episode[:t]]:
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]
                policy[state] = np.argmax(Q[state])
    return policy, Q
