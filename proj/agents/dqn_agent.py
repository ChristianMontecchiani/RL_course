import sys, os
sys.path.insert(0, os.path.abspath(".."))
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
from common import helper as h

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mlp(in_dim, mlp_dims: List[int], out_dim, act_fn=nn.ReLU, out_act=nn.Identity):
    """Returns an MLP."""
    if isinstance(mlp_dims, int): raise ValueError("mlp dimensions should be list, but got int.")

    layers = [nn.Linear(in_dim, mlp_dims[0]), act_fn()]
    for i in range(len(mlp_dims)-1):
        layers += [nn.Linear(mlp_dims[i], mlp_dims[i+1]), act_fn()]
    # the output layer
    layers += [nn.Linear(mlp_dims[-1], out_dim), out_act()]
    return nn.Sequential(*layers)

class DQNAgent(object):
    def __init__(self, state_shape, n_actions,
                 batch_size=32, hidden_dims=[12], gamma=0.98, lr=1e-3, grad_clip_norm=1000, tau=0.001):
        self.n_actions = n_actions
        self.state_dim = state_shape[0]

        self.policy_net = mlp(self.state_dim, hidden_dims, n_actions).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        
        self.counter = 0

    def update(self, buffer):
        """ One gradient step, update the policy net."""
        self.counter += 1
        # Do one step gradient update
        batch = buffer.sample(self.batch_size, device=device)
        
        # TODO: Task 3: Finish the DQN implementation.
        ########## You code starts here #########
        # Hints: 1. You can use torch.gather() to gather values along an axis specified by dim. 
        #        2. torch.max returns a namedtuple (values, indices) where values is the maximum 
        #           value of each row of the input tensor in the given dimension dim.
        #           And indices is the index location of each maximum value found (argmax).
        #        3.  batch is a namedtuple, which has state, action, next_state, not_done, reward
        #           you can access the value be batch.<name>, e.g, batch.state
        #        4. check torch.nn.utils.clip_grad_norm_() to know how to clip grad norm
        #        5. You can go throught the PyTorch Tutorial given on MyCourses if you are not familiar with it.
        # pass

        # calculate the q(s,a)
        qs = self.policy_net(batch.state).gather(1, batch.action.to(dtype=torch.int64)) # shape: [batch_size, 1]

        # calculate q target
        with torch.no_grad():
            qs_tar = batch.reward + self.gamma * batch.not_done * \
                     self.target_net(batch.next_state).max(dim=1, keepdim=True).values

        loss = F.smooth_l1_loss(qs, qs_tar)
        self.optimizer.zero_grad()
        loss.backward()
        # clip grad norm
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm, error_if_nonfinite=False)
        self.optimizer.step()
        ########## You code ends here #########

        # update the target network
        h.soft_update_params(self.policy_net, self.target_net, self.tau)
        
        return {'loss': loss.item(), 
                'q_mean': qs.mean().item(),
                'num_update': self.counter}

    @torch.no_grad()
    def get_action(self, state, epsilon=0.05):
        # TODO:  Task 3: implement epsilon-greedy action selection
        ########## You code starts here #########
        # pass

        sample = random.random()

        if sample > epsilon:
            if state.ndim == 1:
                state = state[None] # add batch dimension
            state = torch.tensor(state).to(device) # conver state to tensor and put it to device
            q_values = self.policy_net(state)
            return torch.argmax(q_values, dim=1).squeeze().item()
        else:
            return random.randrange(self.n_actions)
        ########## You code ends here #########


    def save(self, fp):
        path = fp/'dqn.pt'
        torch.save({
            'policy': self.policy_net.state_dict(),
            'policy_target': self.target_net.state_dict()
        }, path)

    def load(self, fp):
        path = fp/'dqn.pt'
        d = torch.load(path)
        self.policy_net.load_state_dict(d['policy'])
        self.target_net.load_state_dict(d['policy_target'])