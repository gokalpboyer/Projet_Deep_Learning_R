import numpy as np

class GridWorld:
    def __init__(self, width, height, start, terminal_states, obstacles=None):
        self.width = width
        self.height = height
        self.start = start
        self.terminal_states = terminal_states
        self.obstacles = obstacles if obstacles else []
        self.state = start
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.reward_map = np.zeros((height, width))

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'UP' and x > 0:
            x -= 1
        elif action == 'DOWN' and x < self.height - 1:
            x += 1
        elif action == 'LEFT' and y > 0:
            y -= 1
        elif action == 'RIGHT' and y < self.width - 1:
            y += 1

        if (x, y) in self.obstacles:
            x, y = self.state  # No movement if obstacle

        self.state = (x, y)
        reward = self.reward_map[x, y]
        done = self.state in self.terminal_states

        return self.state, reward, done

    def set_rewards(self, reward_dict):
        for state, reward in reward_dict.items():
            self.reward_map[state[0], state[1]] = reward

    def render(self):
        grid = np.zeros((self.height, self.width), dtype=str)
        grid[:] = ' '
        for (x, y) in self.obstacles:
            grid[x, y] = 'X'
        for (x, y) in self.terminal_states:
            grid[x, y] = 'T'
        x, y = self.state
        grid[x, y] = 'A'
        for row in grid:
            print(' '.join(row))
        print()

    def print_policy(self, policy):
        direction_mapping = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        for x in range(self.height):
            for y in range(self.width):
                if (x, y) in self.obstacles:
                    print('X', end=' ')
                elif (x, y) in self.terminal_states:
                    print('T', end=' ')
                else:
                    print(direction_mapping[policy[x, y]], end=' ')
            print()
        print()
