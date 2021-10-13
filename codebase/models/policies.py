import numpy as np

class EpsilonGreedyPolicy:
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self._Q = Q
        self._epsilon = epsilon

    def set_Q(self, Q):
        self._Q = Q

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        if np.random.random() <= self._epsilon:
            return np.random.randint(self._Q.shape[1])
        return np.argmax(self._Q[obs])

    def __call__(self, obs):
        return self.sample_action(obs)

class GreedyPolicy():
    def __init__(self, Q):
        self._Q = Q

    def set_Q(self, Q):
        self._Q = Q

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        return np.argmax(self._Q[obs])

    def __call__(self, obs):
        return self.sample_action(obs)
