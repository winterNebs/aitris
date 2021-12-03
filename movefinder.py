import numpy as np
from tetromino import Tetromino


# Move finder,
# Takes current state as input and finds valid placements
# This could be changed to find all placements, not just ones accessible from hard dropping
class finder:

    @staticmethod
    def decode_mino(mino):
        for i in range(len(mino)):
            if mino[i] == 1:
                return Tetromino.shape_int[i+1]

    def __init__(self, data):
        self.data = data
        self.board = data[:22, :10]
        self.current_piece = finder.decode_mino(data[28][:7])
        self.simplify()

    size_data = {
        'i': [4, 1],
        'o': [2],
        't': [3, 2, 3, 2],
        's': [3, 2],
        'z': [3, 2],
        'j': [3, 2, 3, 2],
        'l': [3, 2, 3, 2],
        'n': [0]
    }

    # Calculate all valid hard drop placements
    def hard_drops(self):
        '''[0,0,0,x,x,x,x,0,0,0] <- i-0'''      # Start: -3, +7 # Length 4, orientation 0 #
        '''[0,0,0,0,0,x,0,0,0,0] <- i-1'''      # Start: -5, +10# Length 1, orientation 1
        '''[0,0,0,x,x,x,0,0,0,0] <- stjlt-02'''  # Start: -3, +8 # Length 3, orientation 2 #
        '''[0,0,0,0,x,x,0,0,0,0] <- o,stjlt-1 '''# Start: -4, +9 #
        '''[0,0,0,x,x,0,0,0,0,0] <- stjlt-3'''   # Start: -3, +9 #                         #
        moves = []
        for orient in range(len(self.size_data[self.current_piece])):
            # Create array of rotations
            # (6 is the action to rotate)
            # Orient is a number between 0-3
            # Rotates would contain [] for no rotation, [6] for one, [6,6] for rotating twice etc.
            rotates = [6] * orient
            offset = 0

            # calculate the number of shifts needed for the current piece
            shifts = 11 - self.size_data[self.current_piece][orient]
            if (orient == 0 or orient == 2 or (self.current_piece in 'stjlt' and orient == 3)) and self.current_piece != 'o':
                offset = -3
            elif self.current_piece == 'i' and orient == 1:
                offset = -5
            else:
                offset = -4
            # enumerate all possible shifts, (offset because the piece is centered.
            # For each, append the number of shifts left/right needed (1 for left, 2 for right)
            # Append a 0 at the end to drop the piece
            for x in range(shifts):  # Left = 1, Right = 2
                loc = x + offset
                if loc < 0:
                    moves.append(rotates + [1 for i in range(abs(loc))] + [0])
                else:
                    moves.append(rotates + [2 for i in range(loc)] + [0])
        return moves

    # Used to calculate the max amount of down inputs required to place a piece.
    # Not relevant for now.
    def simplify(self):
        depth = []  # Bigger == deeper
        for x in range(len(self.board[0])):
            depth.append(0)
            for y in range(len(self.board)):
                if self.board[y][x] != 1:
                    if y > depth[x]:
                        depth[x] = y
                else:
                    break
        maxdepth = np.amax(depth)
        board = np.array(self.board[:maxdepth+1])
        if all(np.array(board).flatten() == 0):
            return
        board = np.append(np.zeros((21-maxdepth,10)), board, axis=0)
        self.data = np.append(board, self.data[22:], axis=0)
'''
current_piece = 't'
size_data = {
    'i': [4, 1],
    'o': [2],
    't': [3, 2, 3, 2],
    's': [3, 2],
    'z': [3, 2],
    'j': [3, 2, 3, 2],
    'l': [3, 2, 3, 2],
    'n': [0]
}

moves = []
rotates = []
for orient in range(len(size_data[current_piece])):
    rotates.append(6)
    offset = 0
    shifts = 11 - size_data[current_piece][orient]
    if (orient == 0 or orient == 2 or (current_piece in 'stjlt' and orient == 3)) and current_piece != 'o':
        offset = -3
    elif current_piece == 'i' and orient == 1:
        offset = -5
    else:
        offset = -4
    for x in range(shifts):  # Left = 1, Right = 2
        loc = x + offset
        if loc < 0:
            moves.append(rotates + [1 for i in range(abs(loc))] + [0])
        else:
            moves.append(rotates + [2 for i in range(loc)] + [0])
print(len(moves))
'''