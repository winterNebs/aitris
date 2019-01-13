from tetris import *
import movefinder as mf

class wrapper:
    action_space = 34
    #action_space = 10
    def act_hd(self, action):
        finder = mf.finder(self.tetris.output_data())
        hds = finder.hard_drops()
        self.tetris.input_c(hds[action % len(hds)])
        #self.tetris.input_c([action])

    def reset_tetris(self, c=None, seed=1):
        self.tetris = Tetris(c, seed)
        return self.tetris

    def __init__(self, c=None, seed=1):
        self.reset_tetris(c, seed)

    def output_data(self):
        #finder = mf.finder(self.tetris.output_data())
        #return finder.data
        return self.tetris.output_data()
