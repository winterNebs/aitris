from tetris import *
import movefinder as mf
import numpy as np

class wrapper:
    action_space = 34
    #action_space = 10
    def act_hd(self, action):
        finder = mf.finder(self.tetris.output_data())
        hds = finder.hard_drops()
        self.tetris.input_c(hds[action % len(hds)])
        #self.tetris.input_c([action])
        # return next state, reward, done
        return np.array(self.output_data()), self.tetris.reward(), not self.tetris.active

    def reset_tetris(self, c=None, seed=1):
        self.tetris = Tetris(c, seed)
        return self.tetris

    def __init__(self, c=None, seed=1):
        self.tetris = Tetris(c, seed)
        self.reset_tetris(c, seed)

    def output_data(self):
        #finder = mf.finder(self.tetris.output_data())
        #return finder.data
        return [[1 if y > 0 else 0 for y in x] for x in self.tetris.output_data()]
