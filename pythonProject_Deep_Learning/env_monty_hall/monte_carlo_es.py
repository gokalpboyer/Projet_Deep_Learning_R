import numpy as np
import random
from collections import defaultdict

class MonteCarloESAgent:
    def __init__(self, env, num_episodes=1000, gamma=0.9):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.returns = defaultdict(list)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.policy = {state: np.random.choice(env.actions(state)) if env.actions(state) else None for state in env.states()}

        # Initialize Q-values for all state-action pairs
        for state in env.states():
            for action in env.actions(state):
                self.Q[state][action] = 0.0

    def learn(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_history = []
            done = False
            while not done:
                action = self.policy[state]
                if action is None:
                    break
                next_state, reward, done = self.env.step(action)
                episode_history.append((state, action, reward))
                state = next_state

            G = 0
            for t in reversed(range(len(episode_history))):
                state, action, reward = episode_history[t]
                G = self.gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode_history[:t]]:
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
                    self.policy[state] = max(self.Q[state], key=self.Q[state].get) if self.Q[state] else None

    def extract_policy(self):
        return self.policy
