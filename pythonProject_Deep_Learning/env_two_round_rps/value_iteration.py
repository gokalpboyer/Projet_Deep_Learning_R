import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-5):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {state: 0 for state in ['start', 'chosen', 'result']}
        self.policy = {state: None for state in ['start', 'chosen', 'result']}

    def value_iteration(self):
        while True:
            delta = 0
            for state in self.V:
                if state == 'result':
                    continue
                v = self.V[state]
                max_value = float('-inf')
                best_action = None
                for action in self.env.available_actions(state):
                    next_state, reward, done = self.env.step(action)
                    value = reward + self.gamma * (0 if done else self.V[next_state])
                    if value > max_value:
                        max_value = value
                        best_action = action
                self.V[state] = max_value
                self.policy[state] = best_action
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break

    def extract_policy(self):
        return self.policy
