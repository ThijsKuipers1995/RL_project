from typing import Tuple
import gym
import numpy as np
import sys
from .stochasticEnv import stochasticEnv, rng

RIGHT = 0
DOWN = 3
LEFT = 2
UP = 1


class GridworldEnv(stochasticEnv):
    """
    Gym Gridworld environment with deterministic negative rewards for landing
    on a block. Starting from the first row, at each row the reward is
    lowered by 0.5. The starting position is at (0,0) (upper left), the goal
    is at the lower right. From each block there are four different actions to
    go to: Left, right, up and down. For the down direction there are a variable
    number of actions which lead to down, for the other directions there is only
    1.

    Attributes:
        shape: (height, width)
            Describes the dimensions of the grid.
        mu: float
            The base reward.
        sigma: float
            the deviation in reward (this is used for stochastic versions)4
        n_paths: int
            Number of paths (actions) which lead to the down direction.
    """

    def _limit_coordinates(self, coord: list) -> list:
        """
        Ensures that the coordinates are within the environment shape.
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current: list, delta: list, dir: int) -> list:
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (self.shape[0] - 1, self.shape[1] - 1)
        return [(1.0, new_state, lambda: self.mu - new_position[0] / 2, is_done)]

    def __init__(
        self,
        shape: tuple = (4, 7),
        mu: float = -1.0,
        sigma: float = 0.01,
        n_paths: int = 10,
    ) -> None:

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
            P[s] = {a: [] for a in range(nA)}
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], RIGHT)
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], UP)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], LEFT)
            for i in range(self.n_paths):
                P[s][DOWN + i] = self._calculate_transition_prob(position, [1, 0], DOWN)

        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((0, 0), self.shape)] = 1.0

        super(GridworldEnv, self).__init__(nS, nA, P, isd)


class GridworldLeftRightEnv(GridworldEnv):
    """
    Gym gridworld equivalent of the Left-Right problem on a block.
    Starting from the first row, at each row the base reward mu is
    lowered by 0.5. The starting position is at (0,0) (upper left), the goal
    is at the lower right. From each block there are four different actions to
    go to: Left, right, up and down. For the down direction there are a variable
    number of actions which lead to down, for the other directions there is only
    1. The rewards for the left, up and down actions are moddeled by a Gaussian
    with mean of mu and standard deviation of 10. The right action is has a
    standard deviation of sigma.

    Attributes:
        shape: (height, width)
            Describes the dimensions of the grid.
        mu: float
            The base reward.
        sigma: float
            the standard deviation in reward of the right action.
        n_paths: int
            Number of paths (actions) which lead to the down direction.
    """

    def _calculate_transition_prob(self, current: list, delta: list, dir: int) -> list:
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (self.shape[0] - 1, self.shape[1] - 1)
        return [
            (
                1.0,
                new_state,
                lambda: rng.normal(
                    self.mu - new_position[0] / 2, 10 if dir else self.sigma
                ),
                is_done,
            )
        ]


class GridworldStochasticEnv(GridworldEnv):
    """
    Gym Gridworld environment with stochastic negative rewards for landing
    on a block. Starting from the first row, at each row the base reward is
    lowered by 1 and the standard deviation is increased by 1. The starting
    position is at (0,0) (upper left), the goal is at the lower right. From
    each block there are four different actions to go to: Left, right, up and
    down. For each direction there are a variable number of actions which lead
    to that direction. The rewards for all actions are moddeled by a Gaussian
    with mean of mu and standard deviation of 10. The right action is has a
    standard deviation of sigma.

    Attributes:
        shape: (height, width)
            Describes the dimensions of the grid.
        mu: float
            The base reward.
        sigma: float
            the base standard deviation in rewards
        n_paths: int
            Number of paths (actions) for each direction
    """

    def _calculate_transition_prob(self, current: list, delta: list, dir: int) -> list:
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (self.shape[0] - 1, self.shape[1] - 1)
        # return [(1., new_state, lambda: rng.normal(self.mu - new_position[0], new_position[0] + self.sigma), is_done)]
        return [
            (
                1.0,
                new_state,
                lambda: rng.normal(self.mu - new_position[0] / 2, self.sigma),
                is_done,
            )
        ]

    def __init__(
        self,
        shape: tuple = (4, 7),
        mu: float = -1.0,
        sigma: float = 10,
        n_paths: int = 4,
    ) -> None:

        self.shape = shape
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths

        nS = np.prod(self.shape)
        nA = 4 * n_paths

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            for i in range(self.n_paths):
                c = i * 4
                P[s][RIGHT + c] = self._calculate_transition_prob(
                    position, [0, 1], RIGHT
                )
                P[s][UP + c] = self._calculate_transition_prob(position, [-1, 0], UP)
                P[s][LEFT + c] = self._calculate_transition_prob(
                    position, [0, -1], LEFT
                )
                P[s][DOWN + c] = self._calculate_transition_prob(position, [1, 0], DOWN)

        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((0, 0), self.shape)] = 1.0

        super(GridworldEnv, self).__init__(nS, nA, P, isd)


class GridworldReverseStochasticEnv(GridworldStochasticEnv):
    """
    Gym Gridworld environment with stochastic negative rewards for landing
    on a block. Starting from the first row, at each row the base reward
    and the standard deviation are increased by 1. The starting position is at
    (0,0) (upper left), the goal is at the lower right. From each block
    there are four different actions to go to: Left, right, up and down.
    For each direction there are a variable number of actions which lead
    to that direction. The rewards for all actions are moddeled by a Gaussian
    with mean of mu and standard deviation of 10. The right action is has a
    standard deviation of sigma.

    Attributes:
        shape: (height, width)
            Describes the dimensions of the grid.
        mu: float
            The base reward.
        sigma: float
            the base standard deviation in rewards
        n_paths: int
            Number of paths (actions) for each direction
    """

    def _calculate_transition_prob(self, current: list, delta: list, dir: int) -> list:
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (self.shape[0] - 1, self.shape[1] - 1)
        return [
            (
                1.0,
                new_state,
                lambda: rng.normal(
                    self.mu - (self.shape[1] - 1) + new_position[0],
                    new_position[0] + self.sigma,
                ),
                is_done,
            )
        ]
