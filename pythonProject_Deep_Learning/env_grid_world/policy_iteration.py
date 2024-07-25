import numpy as np

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.policy = {state: np.random.choice(env.actions) for state in env.states()}
        self.value_function = np.zeros((env.height, env.width))

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in self.policy.keys():
                if state in self.env.terminal_states or state in self.env.obstacles:
                    continue
                v = self.value_function[state]
                action = self.policy[state]
                self.env.state = state
                next_state, reward, _ = self.env.step(action)
                self.value_function[state] = reward + self.gamma * self.value_function[next_state]
                delta = max(delta, abs(v - self.value_function[state]))
            if delta < 1e-6:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in self.policy.keys():
            if state in self.env.terminal_states or state in self.env.obstacles:
                continue
            old_action = self.policy[state]
            action_values = {}
            for action in self.env.actions:
                self.env.state = state
                next_state, reward, _ = self.env.step(action)
                action_values[action] = reward + self.gamma * self.value_function[next_state]
            new_action = max(action_values, key=action_values.get)
            self.policy[state] = new_action
            if new_action != old_action:
                policy_stable = False
        return policy_stable

    def train(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def extract_policy(self):
        return self.policy
