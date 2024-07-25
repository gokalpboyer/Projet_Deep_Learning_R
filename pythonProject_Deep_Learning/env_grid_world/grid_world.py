import numpy as np

class GridWorld:
    def __init__(self, width, height, start, terminal_states, obstacles=[]):
        self.width = width
        self.height = height
        self.start = start
        self.terminal_states = terminal_states
        self.obstacles = obstacles
        self.state = start
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if action not in self.actions:
            raise ValueError("Invalid action")

        x, y = self.state
        if action == 'UP':
            next_state = (x - 1, y)
        elif action == 'DOWN':
            next_state = (x + 1, y)
        elif action == 'LEFT':
            next_state = (x, y - 1)
        elif action == 'RIGHT':
            next_state = (x, y + 1)

        if next_state in self.obstacles or not (0 <= next_state[0] < self.height and 0 <= next_state[1] < self.width):
            next_state = self.state

        reward = -1 if next_state in self.obstacles else 0
        done = next_state in self.terminal_states

        self.state = next_state

        if done:
            reward = 1

        return next_state, reward, done

    def render(self):
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        for (i, j) in self.obstacles:
            grid[i][j] = 'X'
        for (i, j) in self.terminal_states:
            grid[i][j] = 'T'
        x, y = self.state
        grid[x][y] = 'A'
        for row in grid:
            print(' '.join(row))
        print()

    def states(self):
        return [(i, j) for i in range(self.height) for j in range(self.width) if (i, j) not in self.obstacles and (i, j) not in self.terminal_states]
