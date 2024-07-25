import numpy as np
import random
from collections import defaultdict

class MonteCarloES:
    def __init__(self, env, episodes=2000, gamma=0.9):  # Gamma ajusté pour valoriser les récompenses futures
        self.env = env
        self.episodes = episodes
        self.gamma = gamma
        self.Q = defaultdict(lambda: {action: 0.0 for action in env.actions})
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = {state: random.choice(env.actions) for state in self._get_all_states()}

    def _get_all_states(self):
        return [(i, j) for i in range(self.env.height) for j in range(self.env.width)
                if (i, j) not in self.env.obstacles and (i, j) not in self.env.terminal_states]

    def learn(self):
        for episode_num in range(self.episodes):
            state = self.env.reset()
            episode = []
            done = False
            steps = 0

            while not done and steps < 2000:
                action = random.choice(self.env.actions)  # Exploring Start
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
                print(f"Monte Carlo ES - Episode {episode_num} completed")

    def best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get) if state in self.Q else random.choice(self.env.actions)

    def extract_policy(self):
        """Extracts the policy from the Q-values."""
        policy = {}
        for state in self.Q:
            policy[state] = self.best_action(state)
        return policy
