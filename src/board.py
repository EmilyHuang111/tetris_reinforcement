from copy import deepcopy
import numpy as np
from PIL import Image
import cv2
import torch
from piece import Piece  



class Board:
    """
    self.board is a 2d array of booleans, where self.board[i][j] is true
    if position x = j, y = i has a square that is filled
    self.widths is an array of integers, where self.widths[i] is the
    number of squares at row i
    self.heights is an array of integers, where self.heights[i] is the
    maximum height of any square in column i
    """
    def __init__(self, height=20, width=10, block_size=20):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        self.text_color = (200, 20, 220)
        self.piece_manager = Piece()  # Instantiate the Piece class
        self.reset()
        
        self.colors = self.init_board()
        self.widths = [0] * (self.height + 4)
        self.heights = [0] * self.width

    def init_board(self):
        b = []
        for row in range(self.height + 4):
            row = []
            for col in range(self.width):
                row.append(False)
            b.append(row)
        return b  
        
    def undo(self):
        self.board = self.last_board
        self.colors = self.last_colors
        self.widths = self.last_widths
        self.heights = self.last_heights

    def place(self, x, y, piece):
        
        for pos in piece.body:
            target_y = y + pos[1]
            target_x = x + pos[0]
            # Check bounds
            if (
                target_y < 0
                or target_y >= len(self.board)
                or target_x < 0
                or target_x >= len(self.board[0])
            ):
                raise IndexError(f"Invalid placement: x={target_x}, y={target_y}, board dimensions={len(self.board)}x{len(self.board[0])}")
            
            if self.board[target_y][target_x]:
                raise ValueError(f"Placement conflict at x={target_x}, y={target_y}")



    def drop_height(self, piece, x):
        y = -1
        for i in range(len(piece.skirt)):
            if x + i >= self.width:  # Avoid accessing out-of-bounds columns
                continue
            y = max(self.heights[x + i] - piece.skirt[i], y)
        return y


    def top_filled(self):
        return sum([w for w in self.widths[-4:]]) > 0
        

    def clear_rows(self):
        num = 0
        to_delete = []
        for i in range(len(self.widths)):
            if self.widths[i] < self.width:
                continue
            num += 1
            to_delete.append(i)

        for row in to_delete:
            del self.board[row]
            self.board.append([False] * self.width)

            del self.widths[row]
            self.widths.append(0)

            del self.colors[row]
            self.colors.append([False] * self.width)

        if num > 0:
            heights = []
            for col in range(self.width):
                m = 0
                for row in range(self.height):
                    if self.board[row][col]:
                        m = row + 1
                heights.append(m)
            # print(heights)
            self.heights = heights
        return num 

    
    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.ind, self.piece = self.piece_manager.get_new_piece()
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        return self.compute_board_metrics(self.board)
        
    def get_next_states(self):
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id in [2, 3, 4]:  # S, Z, or I piece
            num_rotations = 2
        else:
            num_rotations = 4
    
        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.adjust_piece_on_collision(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.compute_board_metrics(board)
            curr_piece = self.piece_manager.rotate(curr_piece)  # Use the Piece class's rotate method
        return states

        
    def adjust_piece_on_collision(self, piece, pos):
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

       
    def compute_board_metrics(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes
        
#fix the function names from here 
    
    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        self.ind, self.piece = self.piece_manager.get_new_piece()
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def store(self, piece, pos):
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True, video=None):
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.piece_manager.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render:
                self.render(video)

        self.board = self.store(self.piece, self.current_pos)

        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        return score, self.gameover

    # def render(self, video=None):
    #     if not self.gameover:
    #         img = [self.piece_manager.piece_colors[p] for row in self.get_current_board_state() for p in row]
    #     else:
    #         img = [self.piece_manager.piece_colors[p] for row in self.board for p in row]
    #     img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
    #     img = img[..., ::-1]
    #     img = Image.fromarray(img, "RGB")

    #     img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
    #     img = np.array(img)
    #     img[[i * self.block_size for i in range(self.height)], :, :] = 0
    #     img[:, [i * self.block_size for i in range(self.width)], :] = 0

    #     img = np.concatenate((img, self.extra_board), axis=1)

    #     cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
    #     cv2.putText(img, str(self.score),
    #                 (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

    #     cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
    #     cv2.putText(img, str(self.tetrominoes),
    #                 (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

    #     cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
    #     cv2.putText(img, str(self.cleared_lines),
    #                 (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
    #                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

    #     if video:
    #         video.write(img)

    #     cv2.imshow("Deep Q-Learning Tetris", img)
    #     cv2.waitKey(1)