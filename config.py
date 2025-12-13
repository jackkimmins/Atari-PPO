import torch

# https://ale.farama.org/environments/complete_list/
# ALE/Breakout-v5 - Breakout
# ALE/MsPacman-v5 - Ms. Pac-Man
# ALE/SpaceInvaders-v5 - Space Invaders
# ALE/Pong-v5 - Pong
# ALE/Qbert-v5 - Q*bert
# ALE/Asteroids-v5 - Asteroids
# ALE/Seaquest-v5 - Seaquest

# Environment
ENV_NAME = "ALE/Breakout-v5"
FRAME_STACK = 4
FRAME_SKIP = 4
RESIZE_SHAPE = (84, 84)
STICKY_ACTION_PROB = 0.25

# Training hyperparameters
TOTAL_TIMESTEPS = 500_000_000
NUM_ENVS = 16
NUM_STEPS = 128
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = 256
NUM_EPOCHS = 4
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
ENTROPY_COEF_FINAL = 0.001
MAX_GRAD_NORM = 0.5
ANNEAL_LR = True
ANNEAL_ENTROPY = True

# Evaluation
EVAL_RENDER = True
EVAL_DELAY = 0.01

# Saving and checkpointing
MODEL_PATH = f"models/model_{ENV_NAME[4:]}.pt"
CHECKPOINT_PATH = f"models/model_checkpoint_{ENV_NAME[4:]}.pt"
TENSORBOARD_DIR = "runs/ppo"
RESUME_TRAINING = True

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
SAVE_INTERVAL = 100
