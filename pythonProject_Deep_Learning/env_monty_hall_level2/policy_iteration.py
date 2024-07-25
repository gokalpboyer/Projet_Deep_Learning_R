import numpy as np

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.policy = {state: (np.random.choice(env.get_actions(state)) if env.get_actions(state) else None) for state in env.get_states()}
        self.value_function = {state: 0.0 for state in env.get_states()}

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in self.env.get_states():
                v = self.value_function[state]
                action = self.policy[state]
                if action is not None:
                    next_state, reward, done = self.env.step(action)
                    self.value_function[state] = reward + self.gamma * self.value_function[next_state]
                    self.env.reset()  # Reset environment after each step to maintain consistency
                    delta = max(delta, abs(v - self.value_function[state]))

            if delta < 1e-6:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in self.env.get_states():
            old_action = self.policy[state]
            action_values = {}
            for action in self.env.get_actions(state):
                next_state, reward, done = self.env.step(action)
                action_values[action] = reward + self.gamma * self.value_function[next_state]
                self.env.reset()  # Reset environment after each step to maintain consistency

            if action_values:
                self.policy[state] = max(action_values, key=action_values.get)
            else:
                self.policy[state] = None

            if old_action != self.policy[state]:
                policy_stable = False

        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def extract_policy(self):
        return self.policy
