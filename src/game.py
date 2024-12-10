import argparse
import torch
import pygame
from board import Board
from piece import Piece
from q_learning import QLearning

# Define colors for the game
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def parse_config():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=20, help="Height of the game board")
    parser.add_argument("--width", type=int, default=10, help="Width of the game board")
    parser.add_argument("--block_size", type=int, default=30, help="Size of each block")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path for video output (unused here)")
    parser.add_argument("--model_save_path", type=str, default="trained_models", help="Path to save/load trained models")
    return parser.parse_args()

def test(opt):
    """Run the Tetris game with the AI model."""
    # Set up device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Initialize and load the model
    model = QLearning()
    try:
        model_path = f"{opt.model_save_path}/tetris_2000_state_dict.pth"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return

    # Initialize PyGame
    pygame.init()
    screen_width = opt.width * opt.block_size
    screen_height = opt.height * opt.block_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Tetris AI")
    clock = pygame.time.Clock()

    # Initialize the Tetris board
    env = Board(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    running = True

    while running:
        screen.fill(BLACK)

        # Get the next action using the AI model
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]

        # Apply the action and update the game state
        _, done = env.step(action, render=False)

        # Draw the board and current piece
        draw_board(screen, env, opt.block_size)

        # Refresh the screen
        pygame.display.flip()
        clock.tick(opt.fps)

        if done:
            print("Game over!")
            running = False

    pygame.quit()

def draw_board(screen, board, block_size):
    """Render the Tetris board and pieces on the screen."""
    for row in range(board.height):
        for col in range(board.width):
            if board.board[row][col]:
                pygame.draw.rect(
                    screen,
                    board.colors[row][col],
                    pygame.Rect(
                        col * block_size,
                        (board.height - row - 1) * block_size,
                        block_size,
                        block_size,
                    )
                )
    # Draw grid
    for x in range(0, board.width * block_size, block_size):
        pygame.draw.line(screen, WHITE, (x, 0), (x, board.height * block_size))
    for y in range(0, board.height * block_size, block_size):
        pygame.draw.line(screen, WHITE, (0, y), (board.width * block_size, y))

if __name__ == "__main__":
    opt = parse_config()
    test(opt)
