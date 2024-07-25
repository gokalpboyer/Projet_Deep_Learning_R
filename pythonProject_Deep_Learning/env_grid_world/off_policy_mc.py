import numpy as np
import random
from collections import defaultdict
from base_monte_carlo import BaseMonteCarlo

class OffPolicyMC(BaseMonteCarlo):
    def __init__(self, env, episodes=5000, gamma=0.9, epsilon=0.1):
        super().__init__(env, episodes, gamma, epsilon)
        self.C = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon
        self.behavior_policy = lambda state: random.choice(env.actions)

    def learn(self):
        for episode_num in range(self.episodes):
            state = self.env.reset()
            episode = []
            done = False
            steps = 0

            while not done and steps < 1000:
                action = self.behavior_policy(state)
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
                steps += 1

            G = 0
            W = 1.0
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                self.policy[state] = self.best_action(state)
                if action != self.policy[state]:
                    break
                W *= 1.0 / (self.epsilon / len(self.env.actions) + (1 - self.epsilon) * (action == self.policy[state]))

            if episode_num % 100 == 0:
                print(f"Off-policy MC - Episode {episode_num} completed")
