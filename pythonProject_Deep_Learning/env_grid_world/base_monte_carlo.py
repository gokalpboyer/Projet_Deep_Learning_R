from collections import defaultdict
import numpy as np
import random

class BaseMonteCarlo:
    def __init__(self, env, episodes=1000, gamma=1.0, epsilon=0.1):
        self.env = env
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: {action: 0.0 for action in env.actions})
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = {state: random.choice(env.actions) for state in self._get_all_states()}

    def _get_all_states(self):
        states = []
        for i in range(self.env.width):
            for j in range(self.env.height):
                if (i, j) not in self.env.obstacles:
                    states.append((i, j))
        return states

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return self.best_action(state)

    def best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def extract_policy(self):
        return self.policy
