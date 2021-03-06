# KAE 3/19/2022: Most of this code was used as is from the DDPG code provided in the 
#  Udacity mini-project on the pendulum

# The sections on the multiple agents (didn't realize that was what we were doing; 
#  never saw the ability to do a single agent) were obtained from the following source
#  obtained by searched the internet for "how to handle mutliple agents in ddpg pytorch" 
#  and found the following link: 
# https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#   by Mike Richardson
# Inside that code we have the following header:
""""
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""
# which is provided here for completeness. Each code line / snippet that used information
#  from this source is documented below individually as the rest of the code comes
#  from the Udacity mini-project using DDPG on the pendulum

import numpy as np
import random
import copy
from collections import namedtuple, deque

from replaybuffer import ReplayBuffer
from OUNoise_full import OUNoise
from utilities import hard_update

from ddpg_model import Actor, Critic
#from MYreplay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

# commented out are pretty much as is from the DDPG example
#  the used terms were from the multi-agent help, which acheived 
#  exceeding the training score in just under 120 episodes
# https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
#BATCH_SIZE = 256        # minibatch size
#BATCH_SIZE = 256        # minibatch size
#BATCH_SIZE = 64         # minibatch size
#BATCH_SIZE = 32         # minibatch size
#BATCH_SIZE = 16         # minibatch size
#BATCH_SIZE = 8         # minibatch size
WEIGHT_DECAY = 0        # L2 weight decay
#UPDATE_EVERY = 20        # How often to update the network
#UPDATE_EVERY = 20        # How often to update the network
#UPDATE_EVERY = 2        # How often to update the network
#UPDATE_EVERY = 7        # How often to update the network
#UPDATE_EVERY = 4        # How often to update the network
#UPDATE_EVERY = 6        # How often to update the network
UPDATE_EVERY=10
#UPDATE_EVERY=20
#UPDATE_EVERY=5
UPDATES_PER_STEP=10
#UPDATES_PER_STEP=5
#UPDATES_PER_STEP=20
#UPDATES_PER_STEP=5
#UPDATES_PER_STEP=8
#UPDATES_PER_STEP=15
LEARN_START = 0
#TIMES_UPDATE = 1        # How many times to learn each update
#TIMES_UPDATE = 2        # How many times to learn each update
#TIMES_UPDATE = 5        # How many times to learn each update

## KAE 4/14/2022: These currently do nothing.....
#EPSILON_MIN = 0.1
#EPSILON_MAX = float(1.0)
#EPSILON_DECAY = float(1.0)

#NOISE_MU = 0.0
#NOISE_THETA = 0.15
#NOISE_SIGMA = 0.2

NOISE_MU = 0.0
NOISE_THETA = 0.15
#NOISE_SIGMA = 0.2
NOISE_SIGMA = 0.1 # better than 0.2
#NOISE_SIGMA = 0.05 # really bad
#NOISE_SIGMA = 0.15
#NOISE_SIGMA = 0.075


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# KAE 3/18/2022: This is a modified version of the original ddpg for a single agent,
#  but to allow multiple agents
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
# concerning how to get started on the multiple agents portion
#  the original example made modifications to the OUNoise class to reset with a diminising sigma, but we ignored that 
# we also try to maintain the nomenclature of the various parameters without an s as a 
#  singular value and with an s to be the set of agents of parameters


class DDPGAgent():
    """Interacts with and learns from the environment."""
    memory = None # shared memory for both agents
    def __init__(self, state_size, action_size, num_instance, random_seed, \
        lr_actor=1.0e-4, lr_critic=1.0e-3, tau=1.0e-3, gamma=0.99):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents employed
            random_seed (int): random seed
            lr_actor (float):        learning rate of the actor 
            lr_critic (float):       learning rate of the critic
        """
        # save state, action and number of agent sizes along with our initial seed
        self.state_size = state_size
        self.action_size = action_size
        self.num_instance = num_instance
        self.seed = random.seed(random_seed)
#        self.lr_actor = lr_actor
#        self.lr_critic = lr_critic
        self.tau = tau
        self.gamma = gamma
#        self.epsilon = float(EPSILON_MAX)
#        # KAE 4/13/2022, we'll bring these out if this looks promissing...
#        self.noise_amp = 0.1
#        self.noise_reduction = 0.9999

        # Actor Networks (w/ Target and local Networks)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)

        # Critic Network (w/ Target and local Networks)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)

        # initialize targets same as original networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)
        
        print('DDPG Agent.init, BUFFER_SIZE:',BUFFER_SIZE)
        print('DDPG Agent.init, BATCH_SIZE:',BATCH_SIZE)
        print('DDPG Agent.init, WEIGHT_DECAY:',WEIGHT_DECAY)
        print('DDPG Agent.init, UPDATE_EVERY:',UPDATE_EVERY)
        print('DDPG Agent.init, UPDATES_PER_STEP:',UPDATES_PER_STEP)
        print('DDPG Agent.init, LEARN_START:',LEARN_START)
#        print('DDPG Agent.init, TIMES_UPDATE:',TIMES_UPDATE)
        print('DDPG Agent.init, NOISE_MU:',NOISE_MU)
        print('DDPG Agent.init, NOISE_THETA:',NOISE_THETA)
        print('DDPG Agent.init, NOISE_SIGMA:',NOISE_SIGMA)
#        print('DDPG Agent.init, EPSILON_MIN:',EPSILON_MIN)
#        print('DDPG Agent.init, EPSILON_MAX:',EPSILON_MAX)
#        print('DDPG Agent.init, EPSILON_DECAY:',EPSILON_DECAY)
#EPSILON_MIN = 0.1
#EPSILON_MAX = 1.0
#EPSILON_DECAY = 1.0
#        print('DDPG Agent.init, nosie_amp:',self.noise_amp)
#        print('DDPG Agent.init, noise_reduction:',self.noise_reduction)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, 
                                           weight_decay=WEIGHT_DECAY)
        # Noise process, with num_agents
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#   define the size of the noise to NOW be a tuple of num_agents, action_size) 
#     instead of just action size
# amplitude of OU noise
# this slowly decreases to 0
#        noise = 2.0 # = sigma, but for now leave at default of 0.2
#        noise_reduction = 0.9999 # this would require some kind of reset of the sigma value each timestep, function to be called
        if self.num_instance == 1:
            self.noise = OUNoise(action_size, random_seed, mu=NOISE_MU, theta=NOISE_THETA, sigma=NOISE_SIGMA)
        else:
            self.noise = OUNoise((num_instance, action_size), random_seed, mu=NOISE_MU, theta=NOISE_THETA, sigma=NOISE_SIGMA)

        # Replay memory, shared between all DDPGAgents....
        if DDPGAgent.memory == None:
            DDPGAgent.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
#        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.t_step = 0

    def step(self, states, actions, rewards, 
             next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # KAE 3/18/2022: for some strange reason get list indices must be integers or slices, not tuple python
        #  error message at this point AT THE COMMENTS???, appeared to be due to having the buffer class in here; but perhaps just not closing the environment, clearing restart,
        #  so removed it
        # KAE 3/18/2022: this is the key area where we have to read in all the agents together and then
        #  add them into our buffer separately
        # Save experience / reward for each agent
        if self.num_instance == 1:
#            DDPGAgent.memory.add(states, actions, rewards, 
#                            next_states, dones)
            self.memory.add(states, actions, rewards, 
                            next_states, dones)
        else:
            for agent in range(self.num_instance):
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#    add each tuple set (state, action, reward, next_state, done) to the memory buffer
                self.memory.add(states[agent,:], actions[agent,:], rewards[agent], 
                            next_states[agent,:], dones[agent])
    

        # Learn, if enough samples are available in memory
        if len(self.memory) > LEARN_START:
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # after some exploration from Nathn1123, we added the t_step abitly to push off on the timesteps as well....
#            if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
#            print('In DDPGAgent.step random sampling, len_memory:',len(DDPGAgent.memory))
                for i in range(UPDATES_PER_STEP):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma) # our learn includes the soft update of the networks....
#                estates, eactions, erewards, enext_states, edones = experiences
#                print('\nIn DDPGAgent.step, type_estates:', type(estates))
#                print('\nIn DDPGAgent.step, size_estates:', 
#                      estates.size(), eactions.size(), erewards.size(), enext_states.size(), edones.size()  )
#        print('In DDPGAgent.step, len_states:',len(states))


    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        # KAE 3/18/2022: looks like we now are bringing in number of agents worth of scores
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#    for each agent get an action from the local (actor) network given the individual states
            if self.num_instance == 1:
#                actions = np.zeros(self.action_size)
                actions = self.actor_local(states).cpu().data.numpy()
            else:
                actions = np.zeros((self.num_instance, self.action_size))
                for agent in range(self.num_instance):
                    actions[agent,:] = self.actor_local(states[agent,:]).cpu().data.numpy()
        self.actor_local.train()
#        add_noise = False
        if add_noise:
            actions += self.noise.sample()
#        print('In DDPGAgent.act, actions_raw:',actions)
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()
        print('In DDPGAgent.reset noise')

    def update_noise(self):
        self.actor_local.update_noise()
        print('In DDPGAgent.update (decay) noise')
        
    def set_noise(self, amp, red, add_weight_noise=False):
        self.actor_local.set_noise(amp, red, add_weight_noise)
        print('In DDPGAgent.set noise, amp: reduction, doit',amp, red, add_weight_noise)
        
    def set_snoise(self, amp, red, add_state_noise=False):
        self.actor_local.set_snoise(amp, red, add_state_noise)
        print('In DDPGAgent.set noise, amp: reduction, doit',amp, red, add_weight_noise)
        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + ?? * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # KAE 4/12/22: added by Norm1123 but not by my second solution, leavinig it in
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)  
        
#        print(type(self.epsilon), type(EPSILON_DECAY), type(EPSILON_MIN))
#        self.epsilon = np.max([self.epsilon * EPSILON_DECAY,EPSILON_MIN])
#        print('In DDPGAgent.learn, len_experiences, s, a, r, ns, d:',
#              len(experiences),states.size(),actions.size(),rewards.size(),next_states.size(),dones.size())

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
#        icnt=0
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#            print('In DDPGAgent.soft_update loop type target_param, cnt: ', type(target_param.data), icnt )
#            print('In DDPGAgent.soft_update loop type target_param, cnt: ', target_param.data.size(), icnt )
#            icnt += 1
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
#        print('In DDPGAgent.soft_update len_target_params, local: ',len(target_param.data),len(local_param.data))
#        print('In DDPGAgent.soft_update len_target_params, local: ',type(target_param.data),type(local_param.data))
#        print('In DDPGAgent.soft_update shape target_params, local: ',target_param.data.shape(),local_param.data.shape())
#        print('In DDPGAgent.soft_update type target_model.parameters(): ',type(target_model.parameters()) )
#        print('In DDPGAgent.soft_update len target_model.parameters(): ',dir(target_model.parameters()) )
#        print('In DDPGAgent.soft_update dir target_params, local: ',dir(target_param.data))

# KAE 4/11/2022: NOTE: these functions were part of the original DDPGAgent class
#   but for noise=0 (as suggested), these just become 
#  self.actor(obs.to(device) => self.local_actor(obs.to(device))
#   and self.target_actor(obs.to(device) => self.target_actor(obs.to(device))
#    def act(self, obs, noise=0.0):
#        obs = obs.to(device)
#        action = self.actor(obs) + noise*self.noise.noise()
#        return action

#    def target_act(self, obs, noise=0.0):
#        obs = obs.to(device)
#        action = self.target_actor(obs) + noise*self.noise.noise()
#        return action

