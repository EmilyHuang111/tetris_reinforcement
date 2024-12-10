from random import choice
import random
from copy import deepcopy
import numpy as np

RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
INDIGO = (255, 0, 255)
TURQ = (64, 224, 208)

BODIES = [
    (((0, 0), (0, 1), (0, 2), (0, 3)), RED),  # stick 
    
    (((0, 0), (0, 1), (0, 2), (0, 3)), RED),  # stick
    
    (((0, 0), (0, 1), (0, 2), (1, 0)), ORANGE),  # L1 
    (((0, 0), (1, 0), (1, 1), (1, 2)), ORANGE),  # L2 
    (((0, 0), (1, 0), (1, 1), (2, 1)), GREEN),  # S1 
    (((0, 1), (1, 0), (1, 1), (2, 0)), GREEN),  # S2 
    (((0, 0), (0, 1), (1, 0), (1, 1)), TURQ),  # Square 
    (((0, 0), (0, 1), (1, 0), (1, 1)), TURQ),  # Square
    (((0, 0), (1, 0), (1, 1), (2, 0)), CYAN),  # pyramid 
    (((0, 0), (1, 0), (1, 1), (2, 0)), CYAN),  # pyramid
]

BODIES2 = [
    (((0, 0), (0, 1), (0, 2), (0, 3)), RED),  # stick
    (((0, 0), (0, 1), (0, 2), (1, 0)), ORANGE),  # L1
    (((0, 0), (1, 0), (1, 1), (1, 2)), ORANGE),  # L2
    (((0, 0), (1, 0), (1, 1), (2, 1)), GREEN),  # S1
    (((0, 1), (1, 0), (1, 1), (2, 0)), GREEN),  # S2
    (((0, 0), (0, 1), (1, 0), (1, 1)), TURQ),  # Square
    (((0, 0), (1, 0), (1, 1), (2, 0)), CYAN),  # pyramid
]

class Piece:
    """
    A piece is represented with its body and skirt.

    self.body is an array of tuples, where each tuple represents a square in
    the piece's cartesian coordinate system.

    self.skirt is an array of integers, where self.skirt[i] = the minimum height at x = i
    in the piece's cartesian coordinate system.

    Refer to this pdf:
    https://web.stanford.edu/class/archive/cs/cs108/cs108.1092/handouts/11HW2Tetris.pdf
    """
    
    piece_colors = [
        (0, 0, 0), 
        (64, 224, 208), #TURQ
        (0, 255, 255), #CYAN
        (0, 255, 0), #GREEN
        (0, 255, 0), #GREEN
        (255, 0, 0), #RED
        (255, 165, 0), #ORANGE
        (255, 165, 0) #ORANGE
    ]

    pieces = [
        [[1, 1], #square
         [1, 1]],

        [[0, 2, 0], #pyramid
         [2, 2, 2]],

        [[0, 3, 3], #S1
         [3, 3, 0]],

        [[4, 4, 0], #S2
         [0, 4, 4]],

        [[5, 5, 5, 5]], #Stick

        [[0, 0, 6], #L1
         [6, 6, 6]],

        [[7, 0, 0], #L2
         [7, 7, 7]]
    ]

    def __init__(self, body=None, color=None):
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        if body == None:
            self.body, self.color = choice(BODIES)
        else:
            self.body = body
            self.color = color
        self.skirt = self.calc_skirt()
        
        
    def calc_skirt(self):
        skirt = []
        for i in range(4):
            low = 1000
            for b in self.body:
                if b[0] == i:
                    low = min(low, b[1])
            if low != 1000:
                skirt.append(low)
        return skirt

    def get_new_piece(self):
        if not self.bag:
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        piece_id = self.bag.pop()
        piece_shape = [row[:] for row in self.pieces[piece_id]]
        return piece_id, piece_shape

    def get_next_rotation(self):
        width = len(self.skirt)
        # height = max([b[1] for b in self.body])
        new_body = [(width - b[1], b[0]) for b in self.body]
        leftmost = min([b[0] for b in new_body])
        new_body = [(b[0] - leftmost, b[1]) for b in new_body]
        return Piece(new_body, self.color) 

    def rotate(self, piece):
        """Rotate the piece 90 degrees clockwise."""
        num_rows = len(piece)
        num_cols = len(piece[0])
        rotated = [[piece[num_rows - 1 - j][i] for j in range(num_rows)] for i in range(num_cols)]
        return rotated
        

def main():
    for b in BODIES:
        p = Piece(b)
        print(p.skirt)
        
if __name__ == "__main__":
    main()








