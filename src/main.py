from game import Game
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <mode>")
        return

    mode = sys.argv[1]
    agent = None

    if mode == "student":
        from custom_model import CUSTOM_AI_MODEL
        agent = CUSTOM_AI_MODEL()  # Initialize the custom model

    # Initialize the game with the selected mode and agent
    g = Game(mode, agent)
    g.run()
    #g.run_no_visual()

if __name__ == "__main__":
    main()
