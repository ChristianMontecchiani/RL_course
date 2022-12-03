import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np

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



class PPO(object):
    def __init__(self, state_dim, action_dim, lr, batch_size, gamma):
        self.obs_dim = state_dim
        self.act_dim = action_dim
       
        self.gamma = gamma
        self.batch_size = batch_size

        self.actor = Policy(state_dim, action_dim).to(device)
        self.critic = Value(state_dim).to(device)
        
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
    

    def update(self):
        print("Updtating networks..\n")

        self.states = torch.stack(self.states)
        self.dones = torch.stack(self.dones).squeeze()
        self.next_states = torch.stack(self.next_states)
        self.rewards = torch.stack(self.rewards).squeeze()
        self.actions = torch.stack(self.actions).squeeze()
        self.action_log_probs = torch.stack(self.action_log_probs).squeeze()

        for e in range(self.epochs):
            pass
        
        self.dones = list()
        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.next_states = list()
        self.action_logprobs = list()

        print("Update finished!")
    
    def compute_returns(self):
        """
            Compute the Reward-To-Go
        """
        returns = list()
        
        with torch.no_grad():
            state_value = self.critic(self.states)
            next_state_value = self.critic(self.next_states)

            state_value = state_value.squeze()
            next_state_value = next_state_value.squeze()
        
        gaes = torch.zeros()
        start = len(self.rewards) -1 
        end = -1 
        step = -1

        for i in range(start, end, step):
            advs = self.rewards[i] + self.gamma * next_state_value[i] * (1 - self.dones[i]) - state_value[i]
            gaes = advs + self.gamma*self.tau*(1-self.dones[i])*gaes
            
        return torch.Tensor(list(reversed(returns))) 
    
    def get_action(self, obs):
        
        mean = self.actor(obs)
        distribution = Normal(mean, self.covariance_matrix)

        action = distribution.sample()
        log_prob = distribution.log_prob()

        
        return action.detach(), log_prob.detach()
    


