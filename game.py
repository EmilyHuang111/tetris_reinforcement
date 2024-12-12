import pygame
from board import Board
from piece import Piece
from custom_model import CUSTOM_AI_MODEL

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Game:
    def __init__(self, mode, agent=None):
        self.board = Board(height=20, width=10, block_size=50)
        self.curr_piece = Piece()        
        self.screenWidth = self.board.width * self.board.block_size
        self.screenHeight = self.board.height * self.board.block_size
        self.block_size = self.board.block_size
        self.mode = mode
        self.top = 0
        self.y = 20
        self.x = 5
        self.pieces_dropped = 0
        self.rows_cleared = 0
        if mode == "student":
            self.ai = CUSTOM_AI_MODEL()
        else:
            self.ai = None
            
    def run_no_visual(self):
        pygame.init()
        self.screenSize = self.screenWidth, self.screenHeight
        self.pieceHeight = (self.screenHeight - self.top) / self.board.height
        self.pieceWidth = self.screenWidth / self.board.width
        clock = pygame.time.Clock()
        running = True
        render_interval = 500 if not self.ai else 100  # Faster updates for AI

        pygame.time.set_timer(pygame.USEREVENT, render_interval)
        
        if self.ai != None:
            MOVEEVENT, t = pygame.USEREVENT + 1, 100
        #    print('AI')
        else:
            MOVEEVENT, t = pygame.USEREVENT + 1, 500

        while running:
            x, bestPiece = self.ai.get_best_move(self.board,self.curr_piece)
            self.drop(0, x=x)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if self.ai and event.type == pygame.USEREVENT:
                    # Use the AI to decide the best move
                    action,bestPiece = self.ai.get_best_move(self.board,self.curr_piece)  
                    score, gameover,lineCleared,self.curr_piece = self.board.move(action)
                    print('Line cleared',lineCleared)
            clock.tick(30)

        pygame.quit()


    def run(self):
        pygame.init()
        self.screenSize = self.screenWidth, self.screenHeight
        self.pieceHeight = (self.screenHeight - self.top) / self.board.height
        self.pieceWidth = self.screenWidth / self.board.width
        self.screen = pygame.display.set_mode(self.screenSize)
        screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption("Tetris Reinforcement Learning Project")
        clock = pygame.time.Clock()
        running = True
        if self.ai != None:
            MOVEEVENT, t = pygame.USEREVENT + 1, 100
            print('AI')
        else:
            MOVEEVENT, t = pygame.USEREVENT + 1, 500

        pygame.time.set_timer(MOVEEVENT, t)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if self.ai and event.type == pygame.USEREVENT:
                    # Use the AI to decide the best move
                    action,bestPiece = self.ai.get_best_move(self.board,self.curr_piece)  
                    score, gameover,lineCleared,self.curr_piece = self.board.move(action)
                    self.draw_board(screen)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        y = self.board.drop_height(self.board.piece_manager.piece, self.x)
                        self.drop(y)
                        if self.board.top_filled():
                            running = False
                            break
                    if event.key == pygame.K_a:
                        if self.x - 1 >= 0:
                            self.x -= 1
                    if event.key == pygame.K_d:
                        if self.x + 1 <= self.board.width - len(self.board.piece_manager.piece.skirt):
                            self.x += 1
                    if event.key == pygame.K_w:
                        self.board.piece_manager.piece = self.board.piece_manager.piece.get_next_rotation()

                if event.type == MOVEEVENT:
                    if not self.board.verify_collision(self.board.piece, self.board.current_pos):                        
                        # Get the best move and perform rotations
                        action, bestPiece = self.ai.get_best_move(self.board, self.curr_piece)
                        x, num_rotations = action

                        for _ in range(num_rotations):
                            self.board.piece = self.board.piece_manager.rotate(self.board.piece)
                        
                        # Move the active piece down one row
                        self.board.current_pos["y"] += 1

                        # Ensure x-coordinate is retained if no collision occurred
                        if not self.board.verify_collision(self.board.piece, self.board.current_pos):
                            self.board.current_pos["x"] = x  # Preserve the x-coordinate
                    else:
                        # If there's a collision, finalize the placement and generate a new piece
                        action, bestPiece = self.ai.get_best_move(self.board, self.curr_piece)
                        score, gameover, lineCleared, self.curr_piece = self.board.move(action)
                        
                        # Preserve the final x-coordinate after collision resolution
                        x, num_rotations = action
                        self.board.current_pos = {"x": x, "y": 0}
                        for _ in range(num_rotations):
                            self.board.piece = self.board.piece_manager.rotate(self.board.piece)
                        
                        self.draw_board(screen)

            # Render the game
            screen.fill(BLACK)
            self.draw_grid()
            self.draw_board(screen)
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        print(f"Game Over! Score: {self.board.score}, Lines Cleared: {self.board.cleared_lines}")

    def draw_grid(self):
        for row in range(0, self.board.height):
            start = (0, row * self.pieceHeight + self.top)
            end = (self.screenWidth, row * self.pieceHeight + self.top)
            pygame.draw.line(self.screen, WHITE, start, end, width=2)
        for col in range(1, self.board.height):
            start = (col * self.pieceWidth, self.top)
            end = (col * self.pieceWidth, self.screenHeight)
            pygame.draw.line(self.screen, WHITE, start, end, width=2)
        # border
        tl = (0, 0)
        bl = (0, self.screenHeight - 2)
        br = (self.screenWidth - 2, self.screenHeight - 2)
        tr = (self.screenWidth - 2, 0)
        pygame.draw.line(self.screen, WHITE, tl, tr, width=2)
        pygame.draw.line(self.screen, WHITE, tr, br, width=2)
        pygame.draw.line(self.screen, WHITE, br, bl, width=2)
        pygame.draw.line(self.screen, WHITE, tl, bl, width=2)

    def draw_board(self, screen):
        # Get the dynamic board state, including the active piece
        dynamic_board = self.board.retrieve_board_state()
        # Iterate over the board and draw each cell
        for y, row in enumerate(dynamic_board):
            for x, cell in enumerate(row):
                if cell != 0:  # Only draw non-empty cells
                    color = self.board.piece_manager.piece_colors[cell]  # Get color for the piece
                    rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                    pygame.draw.rect(screen, color, rect)


    def update_game_state(self):
        # Use the current piece and its position to calculate new states
        states = self.board.get_next_states()
        
        # Determine the best action based on the AI model or player's move
        action = None
        if self.ai:
            action = self.ai.get_best_move(self.board)  # AI will decide the best action

        # If there's an action, we update the board
        if action:
            self.board.move(action)  # Execute the action (move or rotate)

    def drop(self, y, x=None):
        """
        Drops the current piece onto the board at the specified x and y coordinates.
        Captures rows cleared using the check_cleared_rows logic from board.py.

        :param y: The y-coordinate where the piece lands
        :param x: The x-coordinate where the piece lands (defaults to current x)
        """
        if x is None:
            x = self.x

        try:
            # Check and clear rows
            rows_cleared, self.board.board = self.board.check_cleared_rows(self.board.board)
            self.rows_cleared += rows_cleared

            # Reset the current piece and position
            self.x = 5
            self.y = self.board.height  # Reset y to top of the board
            self.curr_piece = Piece()  # Generate a new piece
            self.pieces_dropped += 1
        except IndexError as e:
            print(f"IndexError during placement: {e}")
            self.gameover = True
        except ValueError as e:
            print(f"ValueError during placement: {e}")
            self.gameover = True
