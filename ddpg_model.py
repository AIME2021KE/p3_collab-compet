# KAE 3/19/2022: This is modified with a batch norm layer added and the layer sizes 
#   reduced from the model for the DDPG pendulum example

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

#DEFAULT_FC1_ACTOR = 128
#DEFAULT_FC2_ACTOR = 256
#DEFAULT_FC1_ACTOR = 64
#DEFAULT_FC2_ACTOR = 128
#DEFAULT_FC1_ACTOR = 48
#DEFAULT_FC2_ACTOR = 96
#DEFAULT_FC1_ACTOR = 20
#DEFAULT_FC2_ACTOR = 40
DEFAULT_FC1_ACTOR = 20
DEFAULT_FC2_ACTOR = 20
#DEFAULT_FC1_ACTOR = 32
#DEFAULT_FC2_ACTOR = 64
DEFAULT_FC1_CRITIC = DEFAULT_FC1_ACTOR
DEFAULT_FC2_CRITIC = DEFAULT_FC2_ACTOR
#DEFAULT_FC1_ACTOR = 400
#DEFAULT_FC2_ACTOR = 300
#DEFAULT_FC1_CRITIC = 400
#DEFAULT_FC2_CRITIC = 300

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=DEFAULT_FC1_ACTOR, \
                 fc2_units=DEFAULT_FC2_ACTOR, momentum=0.1):
#    def __init__(self, state_size, action_size, seed, fc1_units=32, fc2_units=64, \
#        momentum=0.1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            momentum (float): value of the momemtum for the normalization layer
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.momentum = momentum # don't think we need it except in init, however
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        # KAE 4/11/2022: suggested by corporate liason J.Poczatek 
        #  to batch norm after first fully connected layer
        #  KAE 4/12/2022: OK getting 2D/3D but got 1D errors 
        #  with batch norm, which is a new addition.
        #  try removing it....
        self.do_batchnorm = False
        if self.do_batchnorm:
            print('Actor BatchNorm1d on, init....')
            self.norm1d = nn.BatchNorm1d(fc1_units, momentum=momentum)
            
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.do_dropout = False
        if self.do_dropout:
#        self.pdrop = 0.25
            self.pdrop = 0.05
#        self.pdrop = 0.00
            self.dropout = nn.Dropout(p=self.pdrop)
    
        self.reset_parameters()
        
        self.add_weight_noise = False
#        self.noise_amp = 0.000001
        self.noise_amp = 0.00000
#        self.noise_amp = 0.0003
#        self.noise_amp = 0.001
        self.noise_reduction = 0.9999
        if self.add_weight_noise:
            print('Actor weight noise on, amp, red:',self.noise_amp, self.noise_reduction)
        
        self.add_state_noise = False
#        self.snoise_amp = 0.0001
#        self.snoise_amp = 0.01
        self.snoise_amp = 0.00
#        self.snoise_amp = 0.1
#        self.snoise_amp = 1.0
#        self.snoise_amp = 1.0
        self.snoise_reduction = 0.9999
        if self.add_state_noise:
            print('Actor state noise on, amp, red:',self.snoise_amp, self.snoise_reduction)
        
#        self.add_noise_to_weights()
#        self.update_noise()
#        self.set_noise() 
        

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def add_noise_to_weights(self, w):
        """Add noise to the local weights.
        Params
        ======
            m: PyTorch model (weights will be copied from)
        """
        with torch.no_grad():
#            print('In ddpg_model.add_wts_noise: ', w.size())
#            for param in m.parameters():
#            for param in model.parameters():
#                if hasattr(param, 'weight'):
#            print('In ddpg_model.add_wts_noise wt size, noiseamp: ', w.size(), self.noise_amp)
            w.add_(torch.randn(w.size()).to(device)*self.noise_amp)
#            print('In DDPGAgent.soft_update loop type target_param, cnt: ', target_param.data.size(), icnt )

    def add_noise_to_states(self, x):
        """Add noise to the local layer outputs.
        Params
        ======
            x: PyTorch model output
        """
        with torch.no_grad():
#            print('In ddpg_model.add_states_noise, x: ', x.size())
            x += torch.randn(x.size()).to(device)*self.snoise_amp
        return x
            
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.add_weight_noise:
            self.add_noise_to_weights(self.fc1.weight)
            self.add_noise_to_weights(self.fc2.weight)
#            self.add_noise_to_weights(self.fc3.weight)

        x = self.fc1(state)            
#        if self.add_state_noise:
#            self.add_noise_to_states(x)
#        if self.do_dropout:
#            x = self.dropout(x)
#        if self.do_batchnorm:
#            x = x.view(-1,self.fc1_units)
#            x = self.norm1d(x)
        x = F.relu(x)
        x = self.fc2(x)
#        if self.do_dropout:
#            x = self.dropout(x)
#        if self.add_state_noise:
#            self.add_noise_to_states(x)
        x = F.relu(x)
        x  = self.fc3(x)
        return F.tanh(x)
    
    def update_noise(self):
        """Add noise to the local weights.
        Params
        ======
            m: PyTorch model (weights will be copied from)
        """
        self.noise_amp *= self.noise_reduction
        self.snoise_amp *= self.snoise_reduction        
        print('In ddpg_model.update_noise: ', self.noise_amp , self.snoise_amp )
        
    def set_noise(self, amp, red, add_weight_noise = False):
        """Add noise to the local weights.
        Params
        ======
            m: PyTorch model (weights will be copied from)
        """
        self.noise_amp = amp
        self.noise_reduction = red
        self.add_weight_noise = add_weight_noise

    def set_snoise(self, amp, red, add_state_noise = False):
        """Add noise to the local weights.
        Params
        ======
            m: PyTorch model (weights will be copied from)
        """
        self.noise_amp = amp
        self.noise_reduction = red
        self.add_state_noise = add_state_noise

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=DEFAULT_FC1_CRITIC, \
                 fc2_units=DEFAULT_FC1_CRITIC, momentum=0.1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            momentum (float): value of the momemtum for the normalization layer
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.momentum = momentum # don't think we need it except in init, however
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        # KAE 4/11/2022: suggested by corporate liason J.Poczatek 
        #  to batch norm after first fully connected layer
        #  KAE 4/12/2022: OK getting 2D/3D but got 1D errors 
        #  with batch norm, which is a new addition.
        #  try removing it....
        
        self.do_batchnorm = False
        if self.do_batchnorm:
            print('Critic BatchNorm1d on, init....')
            self.norm1d = nn.BatchNorm1d(fc1_units, momentum=momentum)
            
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        
        self.do_dropout = False
        if self.do_dropout:
            self.pdrop = 0.25
            self.dropout = nn.Dropout(p=self.pdrop)
        
#        self.add_weight_noise = True
#        self.noise_amp = 0.1
#        self.noise_reduction = 0.9999
        
#        self.add_state_noise = False
#        self.snoise_amp = 0.1
#        self.snoise_reduction = 0.9999

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
#        if self.add_weight_noise:
#            self.add_noise_to_weights(self.fc1.weight)
#            self.add_noise_to_weights(self.fc2.weight)
##            self.add_noise_to_weights(self.fc3.weight)
        xs = self.fc1(state)
#        if self.do_dropout:
#            xs = self.dropout(xs)
#        if self.add_state_noise:
#            self.add_noise_to_states(xs)
#        if self.do_batchnorm:
#            xs = xs.view(-1,self.fc1_units)
#            xs = self.norm1d(xs)
        xs = F.relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
#        if self.do_dropout:
#            x = self.dropout(x)
#        if self.add_state_noise:
#            self.add_noise_to_states(x)
        x = F.relu(x)
        return self.fc3(x)
