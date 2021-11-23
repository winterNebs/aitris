import random
import copy
import numpy as np
from tetromino import *


def v_height():
    return 20


def height():
    return 22


def width():
    return 10


def pieces():
    return ['i', 'o', 't', 's', 'z', 'j', 'l']


def size():
    return 21


class Tetris:
    def __init__(self, c=None, seed=1):
        random.seed(seed)
        np.random.seed(seed)
        self.commands = []
        self.play_field = [[]]
        self.hold = Tetromino('n')
        self.play_field_canvas = c
        self.locked = True
        self.active = True
        self.rotated = False
        self.mini = False
        self.b2b = False
        self.blocks = [[]]
        self.queue = []
        self.queue_display = []
        self.play_field.clear()
        self.blocks.clear()

        self.total_cleared = 0
        self.total_moves = 0
        self.total_pieces = 0

        self.last_score = 0
        self.total_score = 0
        self.current_combo = 0

        self.cleared = 0

        self.input_log = []
        self.current_piece = self.next_piece()

        '''self.score_table = {
            "single" : 100,
            "double" : 300,
            "triple" : 500,
            "tetris" : 800,
            "tsm"    : 100,
            "ts"     : 400,
            "tsms"   : 200,
            "tss"    : 800,
            "tsmd"   : 1200,
            "tsd"    : 1200,
            "tst"    : 1600,
            "b2b"    : 1.5,
            "combo"  : 50,
            "sd"     : 1,
            "hd"     : 2,
            "pc"     : 3000
        }'''
        '''self.score_table = {
            "single": 1000,
            "double": 3000,
            "triple": 5000,
            "tetris": 8000,
            "tsm": 1000,
            "ts": 4000,
            "tsms": 2000,
            "tss": 8000,
            "tsmd": 12000,
            "tsd": 12000,
            "tst": 16000,
            "b2b": 3,
            "combo": 500,
            "sd": 1,
            "hd": 2,
            "pc": 30000
        }'''
        self.score_table = {
            "single": 1,
            "double": 2,
            "triple": 4,
            "tetris": 8,
            "tsm": 1,
            "ts": 4,
            "tsms": 2,
            "tss": 8,
            "tsmd": 12,
            "tsd": 12,
            "tst": 16,
            "b2b": 0,
            "combo": 0,
            "sd": 0,
            "hd": 0,
            "pc": 12
        }
        for j in range(height()):
            self.play_field.append([])
            for i in range(width()):
                self.play_field[j].append(Block(0))
        if self.play_field_canvas is not None:
            for j, row in enumerate(self.play_field):
                self.blocks.append([])
                for i, block in enumerate(row):
                    self.blocks[j].append(self.play_field_canvas.create_rectangle(
                        i * size() + 5 * size(),
                        j * size() + 2 * size(),
                        (i + 1) * size() + 5 * size(),
                        (j + 1) * size() + 2 * size()))

            self.hold_display = self.play_field_canvas.create_polygon(Tetromino.icon['none'])
            for i in range(0, 5):
                self.queue_display.append(self.play_field_canvas.create_polygon(Tetromino.icon[self.queue[i]]))
            self.render()

    def hold_piece(self):
        if self.locked:
            if self.hold.shapeName != 'n':
                self.current_piece, self.hold = Tetromino(self.hold.shapeName), self.current_piece
            else:
                self.hold = self.current_piece
                self.current_piece = self.next_piece()
            self.locked = False
            self.rotated = False

    def next_piece(self):
        while len(self.queue) < 150:  # generate first 150 blocks
            bag = pieces()
            np.random.shuffle(bag)
            self.queue.extend(bag)
            #self.queue.append('t')
        return Tetromino(self.queue.pop(0))

    def render(self):
        for j, row in enumerate(self.play_field):
            for i, block in enumerate(row):
                self.play_field_canvas.itemconfig(self.blocks[j][i], fill=Block.color[Tetromino.shape_int[block.block_type]])
        for i, block in enumerate(self.current_piece.get_coords()):
            self.play_field_canvas.itemconfig(self.blocks[block.y][block.x], fill=Block.color[self.current_piece.shapeName])
        if self.hold.shapeName != 'n':
            self.play_field_canvas.delete(self.hold_display)
            self.hold_display = self.play_field_canvas.create_polygon(Tetromino.icon[self.hold.shapeName])
            self.play_field_canvas.move(self.hold_display, 2*size(), 2*size())
        for i in range(len(self.queue_display)):
            self.play_field_canvas.delete(self.queue_display[i])
            self.queue_display[i] = self.play_field_canvas.create_polygon(Tetromino.icon[self.queue[i]])
            self.play_field_canvas.move(self.queue_display[i], (size() * (width() + 6)), (2 * size() + (size() * i)))

    def rotate(self, direction):
        # Test Rotation Collisions
        # A->B = A - B
        # (Current) - ((Current + Direction) % 4)
        self.current_piece.kick = Point(0, 0)
        self.rotated = True
        # test.rotate(direction)
        for i in range(5):
            test = copy.deepcopy(self.current_piece)
            test.rotate(direction, i)
            fits = True
            for point in test.get_coords():
                if point.x + test.kick.x < 0 or point.x + test.kick.x > width() - 1 \
                        or point.y + test.kick.y < 0 or point.y + test.kick.y > height() - 1 \
                        or self.play_field[point.y + test.kick.y][point.x + test.kick.x].block_type >= 1:
                    fits = False
                    break
            if fits:
                if self.current_piece.shapeName == 't' and i != 4:
                    self.mini = True
                else:
                    self.mini = False
                self.current_piece.rotate(direction, i)  # 1 = CW, -1 = CCW
                return

    def move(self, direction, times=1):
        while times > 0 or times == -1:
            if direction == 1:  # Right
                for point in self.current_piece.get_coords():
                    if point.x == width() - 1 or self.play_field[point.y][point.x + 1].block_type >= 1:
                        return
                self.current_piece.x += 1
            elif direction == 2:  # Down
                for point in self.current_piece.get_coords():
                    if point.y == height() - 1:
                        self.lock()
                        return
                    elif self.play_field[point.y+1][point.x].block_type >= 1:
                        self.lock()
                        return
                if times == -1:
                    self.last_score += self.score_table["hd"]
                else:
                    self.last_score += self.score_table["sd"]
                self.current_piece.y += 1
            elif direction == 3:  # Left
                for point in self.current_piece.get_coords():
                    if point.x == 0 or self.play_field[point.y][point.x - 1].block_type >= 1:
                        return
                self.current_piece.x -= 1
            if times > 0:
                times -= 1
        self.rotated = False

    def lock(self):  # For now unforgiving
        linescleared = set()
        for point in self.current_piece.get_coords():
            if point.y == 10:
                self.active = False
            self.play_field[point.y][point.x] = Block(Tetromino.int_shape[self.current_piece.shapeName])
            line = True
            for block in self.play_field[point.y]:
                if block.block_type == 0:
                    line = False
            if line:
                linescleared.add(point.y)
        self.cleared = len(linescleared)
        if len(linescleared) > 0:
            # print(len(linescleared))
            self.total_cleared += len(linescleared)
            if self.current_piece.shapeName == 't' and self.rotated:
                corners = []
                for point in self.current_piece.get_t_coords():
                    #print(point.x, point.y)
                    corners.append(point.x >= width() or point.x < 0 or point.y >= height() or
                                   self.play_field[point.y][point.x].block_type > 0)
                if corners[0] and corners[1] and (corners[2] or corners[3]): # T-Spin regular
                    if self.b2b:
                        self.last_score += self.score_table[["tss", "tsd", "tst"][len(linescleared)-1]] * self.score_table["b2b"]
                        # print("b2b", len(linescleared), " Tspin!")
                    else:
                        self.last_score += self.score_table[["tss", "tsd", "tst"][len(linescleared)-1]]
                        # print(len(linescleared), " Tspin!")
                    self.b2b = True
                elif (corners[0] or corners[1]) and corners[2] and corners[3]: # Potentially mini
                    if self.mini:
                        if self.b2b:
                            self.last_score += self.score_table[["tsms", "tsmd"][len(linescleared)-1]] * self.score_table["b2b"]
                            # print("b2b", len(linescleared), " mini-tspin!")
                        else:
                            self.last_score += self.score_table[["tsms", "tsmd"][len(linescleared) - 1]]
                            # print(len(linescleared), " mini-tspin!")
                    else:
                        if self.b2b:
                            self.last_score += self.score_table[["tss", "tsd", "tst"][len(linescleared) - 1]] * \
                                               self.score_table["b2b"]
                            #print("b2b", len(linescleared), " Tspin!")
                        else:
                            self.last_score += self.score_table[["tss", "tsd", "tst"][len(linescleared) - 1]]
                            #print(len(linescleared), " Tspin!")
                    self.b2b = True
                else:
                    self.last_score += self.score_table[["single", "double", "triple", "tetris"][len(linescleared)-1]]
                    #print(len(linescleared), " Cleared!")
                    self.b2b = False
            else:
                if self.b2b and len(linescleared) == 4:
                    self.last_score += self.score_table[["single", "double", "triple", "tetris"][len(linescleared)-1]] * self.score_table["b2b"]
                    #print("b2b", len(linescleared), " Cleared!")
                else:
                    self.last_score += self.score_table[["single", "double", "triple", "tetris"][len(linescleared)-1]]
                    #print(len(linescleared), " Cleared!")
                self.b2b = len(linescleared) == 4

            for line in sorted(linescleared):
                self.play_field.insert(0, self.play_field.pop(line))
                for block in self.play_field[0]:
                    block.block_type = 0
            perfect_clear = True
            for j, row in enumerate(self.play_field):
                for i, block in enumerate(row):
                    if block.block_type > 0:
                        perfect_clear = False
            self.current_combo += 1
            #print("combo: ", self.current_combo)
            self.last_score += self.score_table["combo"] * self.current_combo
            if perfect_clear:
                self.last_score += self.score_table["pc"]
                #print("perfect clear")
        else:
            self.current_combo = -1
        self.current_piece = self.next_piece()
        self.locked = True
        self.total_pieces += 1

    def replay(self, i):
        commands = {
            'space': lambda: self.move(direction=2, times=-1),
            'hd': lambda: self.move(direction=2, times=-1),
            'Left': lambda: self.move(direction=3, times=1),
            'l': lambda: self.move(direction=3, times=1),
            'Right': lambda: self.move(direction=1, times=1),
            'r': lambda: self.move(direction=1, times=1),
            'Down': lambda: self.move(direction=2, times=1),
            'sd': lambda: self.move(direction=2, times=1),
            'Shift_L': lambda: self.hold_piece(),
            'h': lambda: self.hold_piece(),
            's': lambda: self.rotate(direction=-1),
            'ccw': lambda: self.rotate(direction=-1),
            'Up': lambda: self.rotate(direction=1),
            'cw': lambda: self.rotate(direction=1),
            'd': lambda: self.double_rotate(),
            '180': lambda: self.double_rotate(),
            'dasr': lambda: self.move(direction=1, times=-1),
            'dasl': lambda: self.move(direction=3, times=-1)
        }
        commands[i]()
        if self.play_field_canvas is not None:
            self.render()

    def run_commands(self, i):
        command_list = i.split(", ")
        while len(command_list) > 0:
            self.input_log.append(command_list[0])
            self.input_c(command_list.pop(0))
        if self.play_field_canvas is not None:
            self.render()

    def double_rotate(self):
        self.rotate(1)
        self.rotate(1)

    def input_c(self, c):
        self.last_score = 0
        for i in range(len(c)):
            if self.active:
                [lambda: self.move(direction=2, times=-1),  # HD    # 0
                 lambda: self.move(direction=3, times=1),   # Left  # 1
                 lambda: self.move(direction=1, times=1),   # Right # 2
                 lambda: self.move(direction=2, times=1),   # SD    # 3
                 lambda: self.hold_piece(),                 # Hold  # 4
                 lambda: self.rotate(direction=-1),         # CCW   # 5
                 lambda: self.rotate(direction=1),          # CW    # 6
                 lambda: self.double_rotate(),              # 180
                 lambda: self.move(direction=1, times=-1),  # DASR
                 lambda: self.move(direction=3, times=-1)][c[0]]()
                self.total_moves += 1
                self.input_log.append(c[0])
            else:
                self.last_score = -self.total_score
            c.pop(0)
        #print(c)
        self.total_score += self.last_score
        #print(self.total_score)
        if self.play_field_canvas is not None:
            self.render()

    def output_data(self):
        data = np.zeros(shape=(29, 10))
        for j, row in enumerate(self.play_field):
            for i, block in enumerate(row):
                data[j][i] = int(block.block_type > 0)
        for i in range(5):
            data[i+22] = Tetromino.one_hot[self.queue[i]]
        data[27] = Tetromino.one_hot[self.hold.shapeName]
        data[28] = Tetromino.one_hot[self.current_piece.shapeName]
        data[28][7] = self.current_piece.shapeOrient/3
        data[28][8] = self.current_piece.x / 10
        data[28][9] = self.current_piece.y / 23
        for i, block in enumerate(self.current_piece.get_coords()):
           data[block.y][block.x] = 0.5
        #print(len(two_pac))
        return data

    def reward(self):
        return self.last_score
