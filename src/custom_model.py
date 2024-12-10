import torch
from q_learning import QLearning  # Assuming QLearning is your model class


class CUSTOM_AI_MODEL:
    def __init__(self, model_path="trained_models/tetris_2000"):
        # Load the pre-trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def get_best_move(self, board, current_piece):
        """
        Determine the best move for the given piece and board state.

        Args:
            board (Board): The current board state.
            current_piece (Piece): The current piece.

        Returns:
            tuple: The best x-coordinate and the rotated piece.
        """
        best_score = float("-inf")
        best_move = None

        # Get all possible next states
        next_states = board.get_next_states()

        for (x, rotation), board_state in next_states.items():
            # Convert board state to a tensor and send it to the model
            board_state = board_state.to(self.device)

            with torch.no_grad():
                score = self.model(board_state).item()

            if score > best_score:
                best_score = score
                best_move = (x, rotation)

        # Extract the best x and rotation, and return the corresponding piece
        x, rotation = best_move
        for _ in range(rotation):
            current_piece = current_piece.get_next_rotation()

        return x, current_piece
