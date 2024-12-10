from game import Game
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [mode]")
        print("Modes: greedy, genetic, mcts, random, student")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "student":
        # Use CUSTOM_AI_MODEL for student mode
        from custom_model import CUSTOM_AI_MODEL
        ai = CUSTOM_AI_MODEL(model_path="trained_models/tetris_2000")
        game = Game(mode="student", agent=ai)
    else:
        # For other modes
        game = Game(mode=mode)

    # Run the game
    game.run_no_visual()


if __name__ == "__main__":
    main()
