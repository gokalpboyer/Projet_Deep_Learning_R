import numpy as np
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, episodes=5000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.q_table = defaultdict(lambda: {action: 0.0 for action in env.actions})
        self.initialize_q_table()

    def initialize_q_table(self):
        for state in self.env.states():
            self.q_table[state] = {action: 0.0 for action in self.env.actions}

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
                if next_state not in self.q_table:
                    self.q_table[next_state] = {a: 0.0 for a in self.env.actions}
                best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
                td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                td_delta = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_delta
                state = next_state

            # Décroître epsilon pour réduire l'exploration au fil du temps
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def extract_policy(self):
        policy = {}
        for state in self.q_table:
            policy[state] = max(self.q_table[state], key=self.q_table[state].get)
        return policy
