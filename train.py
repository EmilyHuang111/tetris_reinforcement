Conversation opened. 2 messages. All messages read.

Skip to content
Using Gmail with screen readers
Enable desktop notifications for Gmail.
   OK  No thanks
5 of 2,223
Files

Huang, Jimin
AttachmentsDec 11, 2024, 12:00 PM (1 day ago)
 

Huang, Jimin <Jimin.Huang@hdrinc.com>
Attachments
Dec 11, 2024, 12:47 PM (1 day ago)
to me

 

 

 8 Attachments
  •  Scanned by Gmail
import argparse
import os
import shutil
from random import random, randint, sample
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from q_learning import QLearning
from board import Board
from piece import Piece


os.environ["QT_QPA_PLATFORM"] = "offscreen"


def parse_config():
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser()

    # Define command-line arguments in a new sequence
    parser.add_argument("--height", type=int, default=20, help="Height of the game board")
    parser.add_argument("--width", type=int, default=10, help="Width of the game board")
    parser.add_argument("--block_size", type=int, default=30, help="Size of each block")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--gamma_factor", type=float, default=0.99, help="Discount factor for reinforcement learning")
    parser.add_argument("--epsilon_start", type=float, default=1, help="Initial value of epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=1e-3, help="Final value of epsilon for exploration")
    parser.add_argument("--decay_epochs", type=float, default=2000, help="Number of epochs to decay epsilon")
    parser.add_argument("--epochs", type=int, default=3000, help="Total number of training epochs")
    parser.add_argument("--save_freq", type=int, default=1000, help="Frequency of saving the model (in epochs)")
    parser.add_argument("--memory_size", type=int, default=30000, help="Replay memory buffer size")
    parser.add_argument("--log_directory", type=str, default="tensorboard", help="Path to save TensorBoard logs")
    parser.add_argument("--model_save_path", type=str, default="trained_models", help="Path to save trained models")

    # Parse arguments and return
    config = parser.parse_args()
    return config



def train(opt, resume_model_path=None, start_epoch=0):
    torch.manual_seed(123)

    # Prepare directories for logs and models
    if os.path.isdir(opt.log_directory) and start_epoch == 0:
        shutil.rmtree(opt.log_directory)
    os.makedirs(opt.log_directory, exist_ok=True)
    os.makedirs(opt.model_save_path, exist_ok=True)

    writer = SummaryWriter(opt.log_directory)
    env = Board(width=opt.width, height=opt.height, block_size=opt.block_size)  # Tetris environment

    # Load the model
    if resume_model_path:
        model.load_state_dict(torch.load(resume_model_path))
        print(f"Resuming training from model: {resume_model_path}")
    else:
        model = QLearning()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = nn.MSELoss()

    # Initialize the environment
    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.memory_size)
    epoch = start_epoch

    while epoch < opt.epochs:
        next_steps = env.get_next_states()
        epsilon = opt.epsilon_end + (max(opt.decay_epochs - epoch, 0) * (
                opt.epsilon_start - opt.epsilon_end) / opt.decay_epochs)
        random_action = random() <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=False)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])

        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue

        if len(replay_memory) < opt.memory_size / 10:
            continue

        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma_factor * prediction
                  for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}/{opt.epochs}, Score: {final_score}, "
              f"Cleared lines: {final_cleared_lines}")
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), f"{opt.model_save_path}/tetris_{epoch}_state_dict.pth")


    torch.save(model.state_dict(), f"{opt.model_save_path}/tetris_final.pth")
    
    print(f"Epoch: {epoch}/{opt.epochs}, Cleared lines: {final_cleared_lines}")


if __name__ == "__main__":
    opt = parse_config()
    opt.epochs = 2500  # Updated total epochs
    train(opt, resume_model_path=None, start_epoch=0)
    
train.txt
Displaying train.txt.
