import sys, os
sys.path.insert(0, os.path.abspath(".."))
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np
from common.helper import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Actor-critic agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)), nn.ReLU(),
            layer_init(nn.Linear(128, 64)), nn.ReLU(),
            layer_init(nn.Linear(64, action_dim), std=0.01), nn.Tanh()
        )
        # TODO: Task 1: Implement actor_logstd as a learnable parameter
        # Use log of std to make sure std doesn't become negative during training
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(state)

        # Make sure action_logstd matches dimension of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)

        # TODO: Task 1: Create a Normal distribution with mean of 'action_mean' and standard deviation of
        #  'action_logstd', and return the distribution
        probs = Normal(action_mean, action_std)

        return probs


class Value(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.value = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1)))
    
    def forward(self, x):
        return self.value(x).squeeze(1)  # output shape [batch,]


class PG_AC(object):
    def __init__(self, state_dim, action_dim, lr, gamma, ent_coeff, normalize=False):
        self.policy = Policy(state_dim, action_dim).to(device)
        self.value = Value(state_dim).to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        if normalize:
            self.scaler = StandardScaler(n_dim=state_dim, device=device)
        else:
            self.scaler = None

        self.gamma = gamma
        self.ent_coeff = ent_coeff

        # a simple buffer
        self.states = None
        self.action_probs = None
        self.action_ents = None
        self.rewards = None
        self.dones = None
        self.next_states = None

        self._reset_buffer()

    def _reset_buffer(self, ):
        self.states = []
        self.action_probs = []
        self.action_ents = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def update(self,):
        action_probs = torch.stack(self.action_probs, dim=0).to(device).squeeze(-1)
        action_ents = torch.stack(self.action_ents, dim=0).to(device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1)
        discounted_rewards = torch.cumsum(rewards * torch.cumprod(self.gamma * torch.ones_like(rewards), dim=-1), dim=-1)
        states = torch.stack(self.states, dim=0).to(device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(device).squeeze(-1)
        dones = torch.stack(self.dones, dim=0).to(device).squeeze(-1)
        # clear buffer
        self._reset_buffer()

        if self.scaler is not None:
            self.scaler.fit(states)
            states = self.scaler.transform(states)
            next_states = self.scaler.transform(next_states)

        values = self.value(states)

        # calculate the target values
        with torch.no_grad():
            next_values = self.value(next_states)
            target_values = discounted_rewards + self.gamma * (1. - dones) * next_values
        
        critic_loss = F.mse_loss(values, target_values)

        # Advantage estimation
        with torch.no_grad():
            adv = (target_values - values)
            adv = (adv - adv.mean()) / adv.std()

        # Compute the optimization term 
        weighted_probs = -action_probs * adv
        actor_loss = torch.mean(weighted_probs)
        loss = critic_loss + actor_loss + self.ent_coeff * action_ents.sum()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {}

    def get_action(self, observation, evaluation=False):
        """Return action (np.ndarray) and logprob (torch.Tensor) of this action."""
        if observation.ndim == 1:
            observation = observation[None]  # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)
        if self.scaler is not None:
            x = self.scaler.transform(x)

        dist = self.policy(x)
        if evaluation:
            action = dist.mean
        else:
            action = dist.sample()

        action_ent = dist.entropy().mean()

        # calculate the log probability of the action
        act_logprob = dist.log_prob(action).sum(-1)

        action, act_logprob = action, act_logprob.squeeze()
        ########## Your code ends here. ###########

        return action, (act_logprob, action_ent)

    def record(self, observation, action_prob, action_ent, reward, done, next_observation):
        self.states.append(torch.tensor(observation, dtype=torch.float32))
        self.action_probs.append(action_prob)
        self.action_ents.append(action_ent)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.dones.append(torch.tensor([done], dtype=torch.float32))
        self.next_states.append(torch.tensor(next_observation, dtype=torch.float32))

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(f'{filepath}/actor.pt'))
        self.value.load_state_dict(torch.load(f'{filepath}/critic.pt'))

    def save(self, filepath):
        torch.save(self.policy.state_dict(), f'{filepath}/actor.pt')
        torch.save(self.value.state_dict(), f'{filepath}/critic.pt')
