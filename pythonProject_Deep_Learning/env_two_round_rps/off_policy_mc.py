import random
from collections import defaultdict

class OffPolicyMC:
    def __init__(self, env, episodes=1000, epsilon=0.1):
        self.env = env
        self.episodes = episodes
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: {action: 0.0 for action in env.actions})
        self.C = defaultdict(lambda: {action: 0.0 for action in env.actions})
        self.target_policy = defaultdict(lambda: random.choice(self.env.actions))
        self.behavior_policy = lambda state: random.choice(self.env.actions)

    def learn(self):
        for episode_num in range(self.episodes):
            episode = []
            state = self.env.reset()
            done = False

            while not done:
                action = self.behavior_policy(state)
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state

            G = 0
            W = 1
            for state, action, reward in reversed(episode):
                G = reward + self.env.gamma * G
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                self.target_policy[state] = self.best_action(state)
                if action != self.target_policy[state]:
                    break
                W = W * 1 / (1 - self.epsilon + (self.epsilon / len(self.env.actions)))

            if episode_num % 100 == 0:
                print(f"Off-policy MC - Episode {episode_num} completed")

    def best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def extract_policy(self):
        policy = {}
        for state in self.Q:
            policy[state] = self.best_action(state)
        return policy
