import random
import numpy as np
from collections import defaultdict


class DynaQAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000, planning_steps=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.planning_steps = planning_steps
        self.q_table = defaultdict(lambda: {action: 0 for action in self.env.actions})
        self.model = defaultdict(lambda: {})

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                if next_state is None:
                    break

                next_action = self.choose_action(next_state)

                # Q-Learning update
                old_value = self.q_table[state][action]
                next_value = max(self.q_table[next_state].values())
                self.q_table[state][action] = (1 - self.alpha) * old_value + self.alpha * (
                            reward + self.gamma * next_value)

                # Model update
                self.model[state][action] = (reward, next_state)

                # Planning (Dyna-Q)
                for _ in range(self.planning_steps):
                    s = random.choice(list(self.model.keys()))
                    a = random.choice(list(self.model[s].keys()))
                    r, s_next = self.model[s][a]
                    if s_next is None:
                        continue
                    next_best_action = max(self.q_table[s_next].values())
                    self.q_table[s][a] = (1 - self.alpha) * self.q_table[s][a] + self.alpha * (
                                r + self.gamma * next_best_action)

                state = next_state

            if episode % 100 == 0:
                print(f"Dyna-Q - Episode {episode} completed")

    def extract_policy(self):
        policy = {}
        for state in self.q_table:
            policy[state] = max(self.q_table[state], key=self.q_table[state].get)
        return policy
