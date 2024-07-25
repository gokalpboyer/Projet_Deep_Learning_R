import random
import numpy as np
from collections import defaultdict

class DynaQAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=50, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.episodes = episodes
        self.Q = defaultdict(lambda: {action: 0.0 for action in env.actions})
        self.model = defaultdict(lambda: {action: (0, 0, False) for action in env.actions})

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return self.best_action(state)

    def best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def learn(self):
        for episode_num in range(self.episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.Q[state][action] += self.alpha * (
                            reward + self.gamma * max(self.Q[next_state].values()) - self.Q[state][action])
                self.model[state][action] = (next_state, reward, done)

                for _ in range(self.planning_steps):
                    s = random.choice(list(self.model.keys()))
                    a = random.choice(list(self.model[s].keys()))
                    s_next, r, d = self.model[s][a]
                    self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[s_next].values()) - self.Q[s][a])

                state = next_state

            if episode_num % 100 == 0:
                print(f"Dyna-Q - Episode {episode_num} completed")

    def extract_policy(self):
        policy = {}
        for state in self.Q:
            policy[state] = self.best_action(state)
        return policy
