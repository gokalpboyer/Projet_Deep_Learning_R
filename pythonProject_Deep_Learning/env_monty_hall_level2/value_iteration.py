import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value_function = {state: 0.0 for state in env.get_states()}
        self.policy = {state: None for state in env.get_states()}

    def value_iteration(self):
        while True:
            delta = 0
            for state in self.env.get_states():
                v = self.value_function[state]
                action_values = []
                for action in self.env.get_actions(state):
                    next_state, reward, done = self.env.step(action)
                    action_value = reward + self.gamma * self.value_function[next_state]
                    action_values.append(action_value)
                    self.env.reset()  # Reset environment after each step to maintain consistency

                self.value_function[state] = max(action_values) if action_values else 0
                delta = max(delta, abs(v - self.value_function[state]))

            if delta < self.theta:
                break

    def extract_policy(self):
        for state in self.env.get_states():
            action_values = {}
            for action in self.env.get_actions(state):
                next_state, reward, done = self.env.step(action)
                action_values[action] = reward + self.gamma * self.value_function[next_state]
                self.env.reset()  # Reset environment after each step to maintain consistency

            self.policy[state] = max(action_values, key=action_values.get) if action_values else None
        return self.policy
