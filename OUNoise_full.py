import numpy as np
import random
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
#    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.8):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        print('OUNoise.reset, mu, sigma, theta, len:',self.mu, self.sigma, self.theta, len(self.state))

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # KAE 3/18/2022: the next line (commented) from the original DDPG code has a problem, self.size is a tuple 
        #   and presumably provides the right addition of terms....
#        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
# help was provided from https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
#   use the 
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        print('OUNoise.sample, len s, x, dx, sigma:',len(self.state),len(x),len(dx),self.sigma)
        return self.state

