import random
from collections import defaultdict

class MonteCarloES:
    def __init__(self, env, episodes=1000, epsilon=0.1):
        self.env = env
        self.episodes = episodes
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: {action: 0.0 for action in env.actions})
        self.returns = defaultdict(lambda: defaultdict(list))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return self.best_action(state)

    def best_action(self, state):
        return max(self.Q[state], key=self.Q[state].get)

    def learn(self):
        for episode_num in range(self.episodes):
            episode = []
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state

            G = 0
            for state, action, reward in reversed(episode):
                G = reward + self.env.gamma * G
                self.returns[state][action].append(G)
                self.Q[state][action] = sum(self.returns[state][action]) / len(self.returns[state][action])

            if episode_num % 100 == 0:
                print(f"Monte Carlo ES - Episode {episode_num} completed")

    def extract_policy(self):
        policy = {}
        for state in self.Q:
            policy[state] = self.best_action(state)
        return policy
