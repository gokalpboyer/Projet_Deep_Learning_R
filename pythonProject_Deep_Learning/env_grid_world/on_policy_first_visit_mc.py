import numpy as np
import random
from collections import defaultdict
from base_monte_carlo import BaseMonteCarlo

class OnPolicyFirstVisitMC(BaseMonteCarlo):
    def __init__(self, env, episodes=5000, gamma=0.9, epsilon=0.1):
        super().__init__(env, episodes, gamma, epsilon)
        self.returns = defaultdict(lambda: defaultdict(list))
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return self.best_action(state)

    def learn(self):
        for episode_num in range(self.episodes):
            state = self.env.reset()
            episode = []
            done = False
            steps = 0  # Limite des étapes

            while not done and steps < 1000:  # Limite à 1000 étapes
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
                steps += 1

            G = 0
            visited_state_action_pairs = set()
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                if (state, action) not in visited_state_action_pairs:
                    visited_state_action_pairs.add((state, action))
                    self.returns[state][action].append(G)
                    self.Q[state][action] = np.mean(self.returns[state][action])
                    self.policy[state] = self.best_action(state)

            if episode_num % 100 == 0:
                print(f"On-policy First Visit MC - Episode {episode_num} completed")

    def best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get) if state in self.Q else random.choice(self.env.actions)
