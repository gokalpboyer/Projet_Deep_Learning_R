import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, discount_factor=0.9, learning_rate=0.1, epsilon=0.1, episodes=1000):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = {state: {action: 0.0 for action in env.actions} for state in ['start', 'chosen', 'revealed']}

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
                best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
                td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.learning_rate * td_error
                state = next_state

    def extract_policy(self):
        policy = {state: max(actions, key=actions.get) for state, actions in self.q_table.items()}
        return policy
