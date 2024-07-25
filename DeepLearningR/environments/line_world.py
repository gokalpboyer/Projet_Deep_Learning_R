import numpy as np


class LineWorld:
    def __init__(self, length=5):
        self.length = length
        self.reset()

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:  # Move left
            self.state = max(0, self.state - 1)
        elif action == 1:  # Move right
            self.state = min(self.length - 1, self.state + 1)

        if self.state == self.length - 1:
            return self.state, 1, True  # Goal reached
        else:
            return self.state, 0, False  # Not yet at the goal

    def render(self):
        print('|' + '-' * self.state + 'X' + '-' * (self.length - self.state - 1) + '|')
