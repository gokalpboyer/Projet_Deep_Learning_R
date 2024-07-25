import numpy as np

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9, theta=0.0001):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value_table = {state: 0 for state in env.states()}
        self.policy = {state: np.random.choice(env.actions(state)) for state in env.states() if env.actions(state)}

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in self.env.states():
                if state not in self.policy:
                    continue
                v = self.value_table[state]
                action = self.policy[state]
                if action:
                    next_state, reward, done = self.env.step(action)
                    if next_state is not None:
                        self.value_table[state] = reward + self.gamma * self.value_table[next_state]
                        delta = max(delta, abs(v - self.value_table[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in self.env.states():
            if state not in self.policy:
                continue
            old_action = self.policy[state]
            action_values = {}
            for action in self.env.actions(state):
                next_state, reward, done = self.env.step(action)
                if next_state is not None:
                    action_values[action] = reward + self.gamma * self.value_table[next_state]
            if action_values:
                best_action = max(action_values, key=action_values.get)
                self.policy[state] = best_action
                if old_action != best_action:
                    policy_stable = False
        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def extract_policy(self):
        return self.policy
