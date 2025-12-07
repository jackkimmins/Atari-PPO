import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


def init_layer(layer, std=np.sqrt(2), bias_val=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_val)
    return layer


# CNN for Atari games based on the Nature DQN paper.
# Three conv layers into a fully connected layer, then splits into a policy head (what action to take) and value head (how good is this state).
class AtariCNN(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        
        # Feature extraction
        self.conv1 = init_layer(nn.Conv2d(4, 32, kernel_size=8, stride=4))
        self.conv2 = init_layer(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_layer(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc = init_layer(nn.Linear(64 * 7 * 7, 512))
        
        # Output heads
        self.policy_head = init_layer(nn.Linear(512, num_actions), std=0.01)
        self.value_head = init_layer(nn.Linear(512, 1), std=1.0)
    
    # Run through the conv layers.
    def _extract_features(self, x):
        x = x / 255.0  # Normalise pixels
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return x
    
    # Returns action logits and value estimate.
    def forward(self, x):
        features = self._extract_features(x)
        return self.policy_head(features), self.value_head(features)
    
    # Sample an action and get its log prob, entropy, and value.
    # If action is given, computes log prob for that action instead.
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)
    
    def get_value(self, x):
        _, value = self(x)
        return value.squeeze(-1)
