import numpy as np
import random
from collections import defaultdict

class OffPolicyMCAgent:
    def __init__(self, env, num_episodes=1000, gamma=0.9):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.Q = defaultdict(lambda: defaultdict(float))
        self.C = defaultdict(lambda: defaultdict(float))
        self.target_policy = {state: np.random.choice(env.actions(state)) if env.actions(state) else None for state in env.states()}

        # Initialize Q-values for all state-action pairs
        for state in env.states():
            for action in env.actions(state):
                self.Q[state][action] = 0.0

    def behavior_policy(self, state):
        actions = self.env.actions(state)
        return random.choice(actions) if actions else None

    def learn(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_history = []
            done = False
            while not done:
                action = self.behavior_policy(state)
                if action is None:
                    break
                next_state, reward, done = self.env.step(action)
                episode_history.append((state, action, reward))
                state = next_state

            G = 0
            W = 1
            for t in reversed(range(len(episode_history))):
                state, action, reward = episode_history[t]
                G = self.gamma * G + reward
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                self.target_policy[state] = max(self.Q[state], key=self.Q[state].get) if self.Q[state] else None
                if action != self.target_policy[state]:
                    break
                W /= 1 / len(self.env.actions(state)) if self.env.actions(state) else 1

    def extract_policy(self):
        return self.target_policy
