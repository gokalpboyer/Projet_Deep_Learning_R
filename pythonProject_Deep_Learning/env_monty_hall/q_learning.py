import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = {state: {action: 0 for action in env.actions(state)} for state in env.states()}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions(state))
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
                td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error
                state = next_state
                if done:
                    break

    def extract_policy(self):
        return {state: max(actions, key=actions.get) for state, actions in self.q_table.items()}
