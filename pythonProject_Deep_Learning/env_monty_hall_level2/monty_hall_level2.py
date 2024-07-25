import numpy as np

class MontyHallEnvLevel2:
    def __init__(self):
        self.states = ['start', 'chosen', 'revealed']
        self.actions = ['A', 'B', 'C', 'stay', 'switch']
        self.reset()

    def reset(self):
        self.state = 'start'
        self.winning_door = np.random.choice(['A', 'B', 'C'])
        self.chosen_door = None
        self.revealed_door = None
        return self.state

    def get_states(self):
        return self.states

    def get_actions(self, state):
        if state == 'start':
            return ['A', 'B', 'C']
        elif state == 'chosen':
            return ['stay', 'switch']
        else:
            return []

    def step(self, action):
        if self.state == 'start':
            self.chosen_door = action
            remaining_doors = [door for door in ['A', 'B', 'C'] if door != self.chosen_door and door != self.winning_door]
            self.revealed_door = np.random.choice(remaining_doors)
            self.state = 'chosen'
            return self.state, 0, False
        elif self.state == 'chosen':
            if action == 'switch':
                self.chosen_door = [door for door in ['A', 'B', 'C'] if door != self.chosen_door and door != self.revealed_door][0]
            self.state = 'revealed'
            reward = 1 if self.chosen_door == self.winning_door else 0
            return self.state, reward, True
        return self.state, 0, False
