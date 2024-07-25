import numpy as np

class MontyHallEnv:
    def __init__(self):
        self.state_list = ['start', 'chosen', 'revealed']
        self.action_list = ['A', 'B', 'C', 'switch', 'stay']
        self.gamma = 0.9
        self.reset()

    def reset(self):
        self.winning_door = np.random.choice(self.action_list[:3])
        self.chosen_door = None
        self.revealed_door = None
        self.state = 'start'
        return self.state

    def step(self, action):
        if self.state == 'start':
            self.chosen_door = action
            remaining_doors = [door for door in self.action_list[:3] if door != self.chosen_door and door != self.winning_door]
            self.revealed_door = np.random.choice(remaining_doors)
            self.state = 'chosen'
            return 'chosen', 0, False
        elif self.state == 'chosen':
            if action == 'switch':
                self.chosen_door = [door for door in self.action_list[:3] if door != self.chosen_door and door != self.revealed_door][0]
            self.state = 'revealed'
            reward = 1 if self.chosen_door == self.winning_door else 0
            return 'revealed', reward, True
        return self.state, 0, False

    def states(self):
        return self.state_list

    def actions(self, state):
        if state == 'start':
            return self.action_list[:3]
        elif state == 'chosen':
            return ['switch', 'stay']
        return []

    def gamma(self):
        return self.gamma
