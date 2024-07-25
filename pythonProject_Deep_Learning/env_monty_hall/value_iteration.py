import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=0.0001):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value_table = {state: 0 for state in env.states()}
        self.policy = {}

    def value_iteration(self):
        while True:
            delta = 0
            for state in self.env.states():
                action_values = []
                for action in self.env.actions(state):
                    next_state, reward, done = self.env.step(action)
                    if next_state is not None:  # Assurez-vous que next_state n'est pas None
                        action_value = reward + self.gamma * self.value_table[next_state]
                        action_values.append(action_value)
                if action_values:
                    max_value = max(action_values)
                    delta = max(delta, abs(max_value - self.value_table[state]))
                    self.value_table[state] = max_value
            if delta < self.theta:
                break

        for state in self.env.states():
            action_values = {}
            for action in self.env.actions(state):
                next_state, reward, done = self.env.step(action)
                if next_state is not None:  # Assurez-vous que next_state n'est pas None
                    action_values[action] = reward + self.gamma * self.value_table[next_state]
            if action_values:
                self.policy[state] = max(action_values, key=action_values.get)

    def extract_policy(self):
        return self.policy
