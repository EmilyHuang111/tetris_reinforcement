import torch
import numpy as np
from copy import deepcopy
from piece import Piece
from board import Board
import os

class CUSTOM_AI_MODEL:
    def __init__(self, model_path="trained_models/tetris_2400"):
        # Load the pre-trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def get_best_move(self, board, piece):
        """
        Determine the best move for the current board state.

        Args:
            board (Board): The current board state.

        Returns:
            tuple: The best (x, rotation) to play the current piece.
        """
        best_score = float("-inf")
        best_move = None
        best_x = -1
        best_piece = None

        # Get all possible next states from the board
        next_states = board.get_next_states()

        for (x, rotation), board_metrics in next_states.items():
            # Send the board metrics to the model
            board_metrics = board_metrics.to(self.device)

            with torch.no_grad():
                score = self.model(board_metrics).item()

            # Update the best move if the current score is better
            if score > best_score:
                best_score = score
                best_move = (x, rotation)

        return best_move,best_piece
