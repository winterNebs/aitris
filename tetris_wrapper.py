from tetris import *
import movefinder as mf
import numpy as np


# Tetris environment wrapper
class wrapper:

    action_space = 34

    # Get next state based on action
    # hd stands for "Hard Drop" which is the act of placing a piece
    def act_hd(self, action):
        # Find valid placements
        finder = mf.finder(self.tetris.output_data())
        hds = finder.hard_drops()
        # Input action based to environment
        self.tetris.input_c(hds[action % len(hds)])

        # return next state, reward, done
        reward = self.tetris.reward()
        if not self.tetris.active:
            reward = -5
        return self.output_formatted_data(), reward, not self.tetris.active

    def reset_tetris(self, c=None, seed=1):
        self.tetris = Tetris(c, seed)
        return self.tetris

    def __init__(self, c=None, seed=1):
        self.tetris = Tetris(c, seed)
        self.reset_tetris(c, seed)

    def output_data(self):
        return self.tetris.output_data()

    def output_formatted_data(self):
        data = np.array(self.output_data())
        concat = np.concatenate((data[:2, :], data[12:-7, :]))
        return concat