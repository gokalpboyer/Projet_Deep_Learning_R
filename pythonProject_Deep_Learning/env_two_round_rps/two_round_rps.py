class TwoRoundRPS:
    def __init__(self):
        self.actions = ['Rock', 'Paper', 'Scissors']
        self.state = 'start'
        self.gamma = 0.9  # Facteur de discount

    def reset(self):
        self.state = 'start'
        return self.state

    def step(self, action):
        if self.state == 'start':
            if action not in self.actions:
                raise ValueError("Invalid action")
            self.state = 'chosen'
            return self.state, 0, False  # next_state, reward, done

        elif self.state == 'chosen':
            if action not in self.actions:
                raise ValueError("Invalid action")
            self.state = 'result'
            reward = self.get_reward(action)
            return self.state, reward, True  # next_state, reward, done

        elif self.state == 'result':
            return None, 0, True  # Terminate the episode

        return self.state, 0, True  # Safety return to prevent NoneType error

    def get_reward(self, action):
        opponent_action = self.opponent_strategy()
        if action == opponent_action:
            return 0  # draw
        elif (action == 'Rock' and opponent_action == 'Scissors') or \
             (action == 'Paper' and opponent_action == 'Rock') or \
             (action == 'Scissors' and opponent_action == 'Paper'):
            return 1  # win
        else:
            return -1  # lose

    def opponent_strategy(self):
        # Simple fixed strategy for the opponent (can be randomized for more complexity)
        return 'Rock'

    def available_actions(self, state):
        if state in ['start', 'chosen']:
            return self.actions
        else:
            return []
