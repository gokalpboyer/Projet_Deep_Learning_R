import random
import numpy as np

class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        q_table = {}
        for state in ['start', 'chosen', 'result']:
            q_table[state] = {action: 0 for action in self.env.actions}
        return q_table

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False

            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.choose_action(next_state)

                if next_state not in self.q_table:
                    self.q_table[next_state] = {a: 0 for a in self.env.actions}

                old_value = self.q_table[state][action]
                next_value = self.q_table[next_state][next_action]

                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_value)
                self.q_table[state][action] = new_value

                state = next_state
                action = next_action

            if episode % 100 == 0:
                print(f"Episode {episode} completed")

    def extract_policy(self):
        policy = {}
        for state in self.q_table:
            policy[state] = max(self.q_table[state], key=self.q_table[state].get)
        return policy
