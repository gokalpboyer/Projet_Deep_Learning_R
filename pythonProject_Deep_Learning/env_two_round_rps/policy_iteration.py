import numpy as np
import random

class PolicyIterationAgent:
    def __init__(self, env, discount_factor=0.9, theta=0.0001):
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.policy = {action: random.choice(env.actions) for action in env.actions}
        self.value_function = {action: 0 for action in env.actions}

    def policy_evaluation(self):
        while True:
            delta = 0
            new_value_function = self.value_function.copy()
            for action in self.env.actions:
                v = self.value_function[action]
                self.env.reset()
                _, reward, _ = self.env.step(action)
                next_action = self.policy[action]
                _, next_reward, _ = self.env.step(next_action)
                new_value_function[action] = reward + self.discount_factor * next_reward
                delta = max(delta, abs(v - new_value_function[action]))
            self.value_function = new_value_function
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for action in self.env.actions:
            old_action = self.policy[action]
            action_values = {}
            self.env.reset()
            _, reward, _ = self.env.step(action)
            for next_action in self.env.actions:
                _, next_reward, _ = self.env.step(next_action)
                action_values[next_action] = reward + self.discount_factor * next_reward
            best_action = max(action_values, key=action_values.get)
            self.policy[action] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    def train(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def extract_policy(self):
        return self.policy
