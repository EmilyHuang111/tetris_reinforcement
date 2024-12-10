import torch
import numpy as np

class CUSTOM_AI_MODEL:
    def __init__(self, model_path="trained_models/tetris_2000"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_model(self):
        try:
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            print(f"Loaded model from {self.model_path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please ensure the path is correct.")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")


    def get_best_move(self, board, piece, depth=1):
        # Prepare the board state as input for the model
        state = torch.FloatTensor(board.get_current_board_state())
        state = state.unsqueeze(0).to(self.device)  # Add batch dimension

        # Get all possible next states
        next_states = board.get_next_states()
        best_move = None
        best_value = -float('inf')

        for (x, rotation), next_state in next_states.items():
            # Predict Q-value for each state using the loaded model
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                q_value = self.model(next_state_tensor)[0, 0].item()

            if q_value > best_value:
                best_value = q_value
                best_move = (x, rotation)

        # Return the best move found
        if best_move:
            x, rotation = best_move
            for _ in range(rotation):
                piece = piece.get_next_rotation()
            return x, piece

        # Fallback: Random move
        print("No valid moves found; defaulting to random move.")
        return randint(0, board.width - 1), piece
