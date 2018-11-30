
class Point:
    def __init__(self, x_pos=0, y_pos=0):
        self.x = x_pos
        self.y = y_pos

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class Tetromino:
    shapes = {
        'n': [[]],
        'i': [[[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]],
              [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]],
              [[0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]]],
        'o': [[[0, 1, 1],
               [0, 1, 1],
               [0, 0, 0]],
              [[0, 0, 0],
               [0, 1, 1],
               [0, 1, 1]],
              [[0, 0, 0],
               [1, 1, 0],
               [1, 1, 0]],
              [[1, 1, 0],
               [1, 1, 0],
               [0, 0, 0]]],
        't': [[[2, 1, 2],
               [1, 1, 1],
               [3, 0, 3]],
              [[3, 1, 2],
               [0, 1, 1],
               [3, 1, 2]],
              [[3, 0, 3],
               [1, 1, 1],
               [2, 1, 2]],
              [[2, 1, 3],
               [1, 1, 0],
               [2, 1, 3]]],
        's': [[[0, 1, 1],
               [1, 1, 0],
               [0, 0, 0]],
              [[0, 1, 0],
               [0, 1, 1],
               [0, 0, 1]],
              [[0, 0, 0],
               [0, 1, 1],
               [1, 1, 0]],
              [[1, 0, 0],
               [1, 1, 0],
               [0, 1, 0]]],
        'z': [[[1, 1, 0],
               [0, 1, 1],
               [0, 0, 0]],
              [[0, 0, 1],
               [0, 1, 1],
               [0, 1, 0]],
              [[0, 0, 0],
               [1, 1, 0],
               [0, 1, 1]],
              [[0, 1, 0],
               [1, 1, 0],
               [1, 0, 0]]],
        'j': [[[1, 0, 0],
               [1, 1, 1],
               [0, 0, 0]],
              [[0, 1, 1],
               [0, 1, 0],
               [0, 1, 0]],
              [[0, 0, 0],
               [1, 1, 1],
               [0, 0, 1]],
              [[0, 1, 0],
               [0, 1, 0],
               [1, 1, 0]]],
        'l': [[[0, 0, 1],
               [1, 1, 1],
               [0, 0, 0]],
              [[0, 1, 0],
               [0, 1, 0],
               [0, 1, 1]],
              [[0, 0, 0],
               [1, 1, 1],
               [1, 0, 0]],
              [[1, 1, 0],
               [0, 1, 0],
               [0, 1, 0]]]
    }
    kick_table = {
        "jlstz": {0: [Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)],
                  1: [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, -2), Point(1, -2)],
                  2: [Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)],
                  3: [Point(0, 0), Point(-1, 0), Point(-1, 1), Point(0, -2), Point(-1, -2)]},
        "i": {0: [Point(0, 0), Point(-1, 0), Point(2, 0), Point(-1, 0), Point(2, 0)],
              1: [Point(-1, 0), Point(0, 0), Point(0, 0), Point(0, -1), Point(0, 2)],
              2: [Point(-1, -1), Point(1, -1), Point(-2, -1), Point(1, 0), Point(-2, 0)],
              3: [Point(0, -1), Point(0, -1), Point(0, -1), Point(0, 1), Point(0, -2)]},
        "o": {0: [Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)],
              1: [Point(0, 1), Point(0, 1), Point(0, 1), Point(0, 1), Point(0, 1)],
              2: [Point(-1, 1), Point(-1, 1), Point(-1, 1), Point(-1, 1), Point(-1, 1)],
              3: [Point(-1, 0), Point(-1, 0), Point(-1, 0), Point(-1, 0), Point(-1, 0)]}
    }
    icon = {
               'i': [0, 0, 40, 0, 40, 10, 0, 10],
               'o': [0, 0, 20, 0, 20, 20, 0, 20],
               't': [0, 10, 10, 10, 10, 0, 20, 0, 20, 10, 30, 10, 30, 20, 0, 20],
               's': [0, 10, 10, 10, 10, 0, 30, 0, 30, 10, 20, 10, 20, 20, 0, 20],
               'z': [0, 0, 20, 0, 20, 10, 30, 10, 30, 20, 10, 20, 10, 10, 0, 10],
               'j': [0, 0, 10, 0, 10, 10, 30, 10, 30, 20, 0, 20],
               'l': [0, 10, 20, 10, 20, 0, 30, 0, 30, 20, 0, 20],
               'none': [0, 0, 0, 0, 0, 0]
    }
    shape_int = {
        0: 'n',
        1: 'i',
        2: 'o',
        3: 't',
        4: 's',
        5: 'z',
        6: 'j',
        7: 'l'
    }
    int_shape = {
        'n': 0,
        'i': 1,
        'o': 2,
        't': 3,
        's': 4,
        'z': 5,
        'j': 6,
        'l': 7
    }

    def __init__(self, c):
        self.shape = Tetromino.shapes[c]
        self.shapeName = c
        self.shapeOrient = 0
        self.x = 3
        self.y = 0
        self.kick = Point(0, 0)
        if self.shapeName == 'i':
            self.x = 2
            self.y = -1

    def get_coords(self):
        points = []
        for j, row in enumerate(Tetromino.shapes[self.shapeName][self.shapeOrient]):
            for i, value in enumerate(row):
                if value == 1:
                    points.append(Point(i + self.x, j + self.y))
        return points

    def get_t_coords(self):
        # [point, point, back, back]
        points = []
        for j, row in enumerate(Tetromino.shapes[self.shapeName][self.shapeOrient]):
            for i, value in enumerate(row):
                if value == 2:
                    points.append(Point(i + self.x, j + self.y))
        for j, row in enumerate(Tetromino.shapes[self.shapeName][self.shapeOrient]):
            for i, value in enumerate(row):
                if value == 3:
                    points.append(Point(i + self.x, j + self.y))
        return points

    def rotate(self, cw, nk):  # CW = 1 CCW = -1
        if self.shapeName in "jlstz":
            self.kick = Tetromino.kick_table["jlstz"][self.shapeOrient][nk] - Tetromino.kick_table["jlstz"][
                ((self.shapeOrient + cw) % 4)][nk]
        else:
            self.kick = Tetromino.kick_table[self.shapeName][self.shapeOrient][nk] - Tetromino.kick_table[self.shapeName][
                ((self.shapeOrient + cw) % 4)][nk]
        self.shapeOrient = (self.shapeOrient + cw) % 4
        self.x += self.kick.x
        self.y += self.kick.y
        self.kick = Point(0, 0)


class Block:
    block_type = 0
    color = {
        'n': "gray",
        'i': "cyan",
        'o': "yellow2",
        't': "purple1",
        's': "green2",
        'z': "firebrick1",
        'j': "blue",
        'l': "chocolate1"
    }

    def __init__(self, t):
        self.block_type = t
