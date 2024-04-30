import numpy as np
import scipy.signal

import torch
# from GCN_t.ddpg.model import GNNPolicy
from ddpg.model import GNNPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=torch.nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j+1]), act()]
    return torch.nn.Sequential(*layers)

# def gnn(sizes, activation, output_activation=torch.nn.Identity):

#     pass

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(torch.nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, torch.nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)
    
class GNNActor(torch.nn.Module):

    def __init__(self, act_limit):
        super().__init__()
        self.pi = GNNPolicy().to(DEVICE)
        self.act_limit = act_limit

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        return self.act_limit * self.pi(constraint_features, edge_indices, edge_features, variable_features)

class MLPQFunction(torch.nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.
    
class GNNQFunction(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.q = GNNPolicy().to(DEVICE)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=1))
        return torch.squeeze(q, -1)
    

class MLPActorCritic(torch.nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=torch.nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
        
class GNNActorCritic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # act_limit = action_space.high[0]
        act_limit = 0.8
        self.pi = GNNActor(act_limit).to(DEVICE)
        self.q = GNNQFunction()

    def act(self, constraint_features, edge_indices, edge_features, variable_features):
        with torch.no_grad():
            return self.pi(constraint_features, edge_indices, edge_features, variable_features).numpy()

