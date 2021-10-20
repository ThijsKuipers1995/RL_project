import gym
import numpy as np
import sys
from .stochasticEnv import stochasticEnv, rng

RIGHT = 0
DOWN = 3
LEFT = 2
UP = 1

class GridworldEnv(stochasticEnv):
    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (self.shape[0]-1, self.shape[1]-1)
        return [(1.0, new_state, lambda: self.mu - new_position[0], is_done)]

    def __init__(self, shape=(4,7), mu=-1.0, sigma=0.01, n_paths=10):
        self.shape = shape
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths

        nS = np.prod(self.shape)
        nA = 3 + n_paths

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], RIGHT)
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], UP)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], LEFT)
            for i in range(self.n_paths):
                P[s][DOWN+i] = self._calculate_transition_prob(position, [1, 0], DOWN)

        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((0,0), self.shape)] = 1.0

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

class GridworldStochasticEnv(GridworldEnv):
    def _calculate_transition_prob(self, current, delta, dir):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (self.shape[0]-1, self.shape[1]-1)
        return [(1., new_state, lambda: rng.normal(self.mu - new_position[0]/2, 10 if dir else self.sigma), is_done)]
