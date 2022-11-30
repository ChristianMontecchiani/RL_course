import sys, os
sys.path.insert(0, os.path.abspath(".."))
import torch
import torch.nn.functional as F
from torch import nn

from torch.distributions import Normal

import numpy as np
from common import helper as h


# Use CUDA for storing tensors / calculations if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Layer initialization func
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# # This class defines the neural network policy
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()

        # Initialise a neural network with two hidden layers (64 neurons per layer)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        std = np.log(np.array([1]))
        #self.actor_logstd = torch.nn.Parameter(torch.from_numpy(std).float().to(device))
        self.actor_logstd = torch.from_numpy(std).float().to(device)
        
    # Do a forward pass to map state to action
    def forward(self, state):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(state)

        # Make sure action_logstd matches dimension of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        return probs


# Class for the Policy Gradient algorithm
class Reinforce(object):
    def __init__(self, state_dim, action_dim, lr, gamma):

        # Define the neural network policy
        self.policy = Policy(state_dim, action_dim).to(device)
        # Create an optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # Set discount factor value
        self.gamma = gamma
        # Simple buffers for action probabilities and rewards
        self.action_probs = []
        self.rewards = []

    def update(self,):
        # Prepare dataset used to update policy
        action_probs = torch.stack(self.action_probs, dim=0).to(device).squeeze(-1) # shape: [batch_size,]
        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1) # shape [batch_size,]
        self.action_probs, self.rewards = [], [] # clean buffers
        
        discounted_rewards = h.discount_rewards(rewards, self.gamma)
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / torch.std(discounted_rewards)
        
        loss = -torch.mul(discounted_rewards, action_probs).sum()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {'logstd': self.policy.actor_logstd.cpu().detach().numpy()}


    def get_action(self, observation, evaluation=False):
        """Return action and logprob of this action."""

        # Add batch dimension if necessary
        if observation.ndim == 1: observation = observation[None]

        # Convert observation to a torch tensor
        x = torch.from_numpy(observation).float().to(device)

        distribution = self.policy(x)
        if evaluation:
            return distribution.mean, None
        
        action = distribution.sample()[0]
        act_logprob = distribution.log_prob(action).sum()
        
        if observation.ndim == 1: action = action[0]

        return action, act_logprob

    def record(self, action_prob, reward):
        """ Store agent's and env's outcomes to update the agent."""
        self.action_probs.append(action_prob)
        self.rewards.append(torch.tensor([reward]))

    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))