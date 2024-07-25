import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, epsilon=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_function = np.zeros((env.height, env.width))

    def value_iteration(self):
        while True:
            delta = 0
            for i in range(self.env.height):
                for j in range(self.env.width):
                    if (i, j) in self.env.terminal_states or (i, j) in self.env.obstacles:
                        continue
                    v = self.value_function[i, j]
                    action_values = []
                    for action in self.env.actions:
                        self.env.state = (i, j)  # Assurez-vous que l'état est défini
                        next_state, reward, _ = self.env.step(action)
                        action_value = reward + self.gamma * self.value_function[next_state]
                        action_values.append(action_value)
                    self.value_function[i, j] = max(action_values)
                    delta = max(delta, abs(v - self.value_function[i, j]))
            if delta < self.epsilon:
                break

    def extract_policy(self):
        policy = {}
        for i in range(self.env.height):
            for j in range(self.env.width):
                if (i, j) in self.env.terminal_states or (i, j) in self.env.obstacles:
                    continue
                action_values = {}
                for action in self.env.actions:
                    self.env.state = (i, j)
                    next_state, reward, _ = self.env.step(action)
                    action_value = reward + self.gamma * self.value_function[next_state]
                    action_values[action] = action_value
                best_action = max(action_values, key=action_values.get)
                policy[(i, j)] = best_action
        return policy
