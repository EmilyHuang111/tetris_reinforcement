def test(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Initialize and load the model
    model = QLearning()  # Ensure this matches the training model
    model.load_state_dict(torch.load("{}/tetris_2000_state_dict.pth".format(opt.saved_path), map_location=device))
    model.to(device)
    model.eval()

    # Initialize the Tetris board
    env = Board(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()

    # Setup video output
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5 * opt.width * opt.block_size), opt.height * opt.block_size))

    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]

        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break

        


if __name__ == "__main__":
    opt = get_args()
    test(opt)
