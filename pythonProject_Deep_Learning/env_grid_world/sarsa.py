import numpy as np
import random


class SARSAAgent:
    def __init__(self, env, discount_factor=0.9, learning_rate=0.1, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 episodes=5000):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.q_table = np.zeros((env.height, env.width, len(env.actions)))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.env.actions)))
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def learn(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            while not done:
                next_state, reward, done = self.env.step(self.env.actions[action])
                next_action = self.choose_action(next_state)
                td_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], next_action]
                td_error = td_target - self.q_table[state[0], state[1], action]
                self.q_table[state[0], state[1], action] += self.learning_rate * td_error
                state = next_state
                action = next_action

            # Décroître epsilon pour réduire l'exploration au fil du temps
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def extract_policy(self):
        policy = np.zeros((self.env.height, self.env.width), dtype=int)
        for x in range(self.env.height):
            for y in range(self.env.width):
                if (x, y) in self.env.terminal_states or (x, y) in self.env.obstacles:
                    policy[x, y] = -1  # Indique un état non navigable
                else:
                    policy[x, y] = np.argmax(self.q_table[x, y])
        return policy
