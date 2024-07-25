import numpy as np
import random
from collections import defaultdict

class DynaQAgent:
    def __init__(self, env, num_episodes=1000, gamma=0.9, alpha=0.1, planning_steps=5):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma  # Facteur de discount
        self.alpha = alpha  # Taux d'apprentissage
        self.planning_steps = planning_steps  # Nombre d'étapes de planification
        self.q_table = defaultdict(lambda: defaultdict(float))  # Table Q
        self.model = defaultdict(lambda: defaultdict(tuple))  # Modèle de l'environnement (modèle de transition)

        # Initialiser les valeurs Q pour toutes les paires état-action
        for state in env.states():
            for action in env.actions(state):
                self.q_table[state][action] = 0.0

    def choose_action(self, state):
        actions = self.env.actions(state)
        if random.uniform(0, 1) < 0.1:  # Paramètre d'exploration (epsilon)
            return random.choice(actions) if actions else None
        return max(self.q_table[state], key=self.q_table[state].get) if actions else None

    def learn(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                if action is None:
                    break
                next_state, reward, done = self.env.step(action)

                # Assurer que les valeurs Q sont initialisées pour next_state
                if next_state not in self.q_table:
                    for next_action in self.env.actions(next_state):
                        self.q_table[next_state][next_action] = 0.0

                # Mettre à jour la valeur Q
                self.q_table[state][action] += self.alpha * (reward + self.gamma * max(self.q_table[next_state].values(), default=0) - self.q_table[state][action])

                # Mettre à jour le modèle
                self.model[state][action] = (next_state, reward)

                # Étape de planification
                for _ in range(self.planning_steps):
                    s = random.choice(list(self.model.keys()))
                    a = random.choice(list(self.model[s].keys()))
                    next_s, r = self.model[s][a]

                    # Assurer que les valeurs Q sont initialisées pour next_s
                    if next_s not in self.q_table:
                        for next_action in self.env.actions(next_s):
                            self.q_table[next_s][next_action] = 0.0

                    # Mettre à jour la valeur Q à partir du modèle
                    self.q_table[s][a] += self.alpha * (r + self.gamma * max(self.q_table[next_s].values(), default=0) - self.q_table[s][a])

                state = next_state

    def extract_policy(self):
        policy = {}
        for state in self.q_table:
            policy[state] = max(self.q_table[state], key=self.q_table[state].get) if self.q_table[state] else None
        return policy
 