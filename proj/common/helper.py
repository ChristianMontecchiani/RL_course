import re
import os
import random
from cv2 import log
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def soft_update_params(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


# credit to https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
def truncated_normal(t, mean=0.0, std=1.0):
    """ Re-drewing the values rather than clipping them."""
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
      cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
      if not torch.sum(cond):
        break
      t = torch.where(cond, torch.nn.init.normal_(torch.ones_like(t), mean=mean, std=std), t)
    return t


def linear_schedule(schdl, step):
    """Outputs values following a linear decay schedule"""
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)


def save_object(obj, filename): 
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
    

def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class StandardScaler:
    """ Used to calculate mean, std and normalize data. """
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        """ Calculate mean and std for given data."""
        assert isinstance(data, torch.Tensor), f"data must be in torch.Tensor, while got {data.dtype}."
        self.mean = data.mean(0, keepdim=True) # calculate mean among batch, shape [1, x_dim]
        self.std = data.std(0, keepdim=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """ Normalization. """
        assert isinstance(self.mean, torch.Tensor), "Call fit() before using transform()."
        assert data.ndim == self.mean.ndim, "mean and data should have the same dimensions."

        if data.device.type == 'cuda':
            mean, std = self.mean.to("cuda"), self.mean.to('cuda')
        else:
            mean, std = self.mean, self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        assert isinstance(self.mean, torch.Tensor), "Call fit() before using inverse_transform()."
        assert data.ndim == self.mean.ndim, "mean and data should have the same dimensions."
        
        if data.device.type == 'cuda':
            mean, std = self.mean.to("cuda"), self.mean.to('cuda')
        else:
            mean, std = self.mean, self.std
        return data * std + mean
    

class NormalizeImg(nn.Module):
    """Module that divides (pixel) observations by 255. and minus 0.5, the value range should be [-0.5, 0.5]."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.) - 0.5


class Flatten(nn.Module):
    """Module that flattens its input to a (batched) vector."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def get_space_dim(space):
    t = type(space)
    if t is gym.spaces.Discrete:
        return space.n
    elif t is gym.spaces.Box:
        return space.shape[0]
    else:
        raise TypeError("Unknown space type:", t)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)