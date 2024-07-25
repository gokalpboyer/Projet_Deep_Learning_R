import random
from collections import defaultdict

class OnPolicyFirstVisitMC:
    def __init__(self, env, episodes=1000, epsilon=0.1):
        self.env = env
        self.episodes = episodes
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: {action: 0.0 for action in env.actions})
        self.returns = defaultdict(lambda: defaultdict(list))
        self.policy = defaultdict(lambda: random.choice(self.env.actions))

    def learn(self):
        for episode_num in range(self.episodes):
            episode = []
            state = self.env.reset()
            done = False

            while not done:
                action = self.policy[state]
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state

            G = 0
            for state, action, reward in reversed(episode):
                G = reward + self.env.gamma * G
                if (state, action) not in [(x[0], x[1]) for x in episode[:-1]]:
                    self.returns[state][action].append(G)
                    self.Q[state][action] = sum(self.returns[state][action]) / len(self.returns[state][action])
                    self.policy[state] = self.best_action(state)

            if episode_num % 100 == 0:
                print(f"On-policy First Visit MC - Episode {episode_num} completed")

    def best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def extract_policy(self):
        policy = {}
        for state in self.Q:
            policy[state] = self.best_action(state)
        return policy
