import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import List, Tuple


LEFT = 0
RIGHT = 1


class LeftRight(gym.Env):
    """
    Source:
        This environment corresponds to the simple Markov Decision Process
        described by Barto, Sutton, and Anderson (Reinforcement Learning -
        An Introduction, p. 134)
    
    Observation:
        Type: Discrete(2)      
        Num     Observation             
        0       State A    
        1       State B
    
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Go to the left
        1     Go to the right
        
        Note: In State B there is technically only one action available, but 
        the behaviour of the environment is unaffected by this. 
    
    Reward:
        Rewards are deterministic in state A (0) and stochastic in state B (1),
        according to the attributes.
    
    Starting State:
        State A (0)
    
    Episode Termination:
        State A, action 1 (going right).
        State B, any action.
    
    Attributes:
        reward_right: Reward of state 0 and action 1 (going right).
        reward_left: Reward of state 0 and action 0 (going left).
        sigma: Standard deviation used for sampling the reward of state 1.
        mu: Mean used for sampling the reward of state 1.
    """

    def __init__(self, 
                 reward_right: float, 
                 reward_left: float,
                 mu: float,
                 sigma: float):
        
        super(LeftRight, self).__init__()
        
        self.reward_right = reward_right 
        self.reward_left = reward_left
        self.sigma = sigma
        self.mu = mu
        
        self.seed()
        self.state = None
        
        self.nS = 2
        self.nA = 2
        
        # Action space (left or right)
        self.action_space = spaces.Discrete(self.nA) 
             
        # Observational space (state 0 or 1)
        self.observation_space = spaces.Discrete(self.nS)
        
    def step(self, action: int) -> Tuple[int, float, bool, dict]:      
        
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        # sample reward from Gaussian distribution when in state 1
        if self.state:
            reward = np.random.normal(loc=self.mu, scale=self.sigma)
            done = True
        # otherwise return deterministic reward for state 0 depending on action
        else:
            if action:
                reward = self.reward_right
                done = True
            else:
                reward = self.reward_left
                done = False
                self.state = 1

        info = {}
        
        return self.state, reward, done, info
    
    def reset(self) -> int:
        self.state = 0 
        return self.state
    
    def seed(self, seed: int=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
