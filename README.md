# Atari Games w/ Proximal Policy Optimisation
This repository contains code and resources for training reinforcement learning agents to play Atari games using the Proximal Policy Optimisation (PPO) algorithm.  Designed for educational purposes, it provides a hands-on approach to understanding RL concepts and techniques.

# How to Use
## 1) Train
To train an agent on a specific Atari game, run the `train.py` script with the desired game environment set in `ENV_NAME` in `config.py`.
Example:
`ENV_NAME = "ALE/Breakout-v5"`

## 2) Test
After training, you can test the trained agent using the `test.py` script. Make sure to move the trained .pt model file to the `fully_trained_models/` directory.