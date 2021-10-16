import numpy as np
from collections import defaultdict

class DynamicEpsilonGreedyPolicy:
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self._Q = Q
        self._epsilon = epsilon
        self.visisted = defaultdict(int)

    def set_Q(self, Q):
        self._Q = Q

    def sample_action(self, obs, env = None):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        self.visisted[obs] += 1

        try:
            actions = env.state_actions(obs)
        except AttributeError:
            actions = self._Q.shape[1]

        if np.random.random() <= self._epsilon / np.sqrt(self.visisted[obs]):
            return np.random.randint(actions)
        return np.argmax(self._Q[obs, :actions])

    def __call__(self, obs):
        return self.sample_action(obs)

class EpsilonGreedyPolicy:
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self._Q = Q
        self._epsilon = epsilon

    def set_Q(self, Q):
        self._Q = Q

    def sample_action(self, obs, env = None):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        try:
            actions = env.state_actions(obs)
        except AttributeError:
            actions = self._Q.shape[1]

        if np.random.random() <= self._epsilon:
            return np.random.randint(actions)
        return np.argmax(self._Q[obs, :actions])

    def __call__(self, obs):
        return self.sample_action(obs)

class GreedyPolicy():
    def __init__(self, Q):
        self._Q = Q

    def set_Q(self, Q):
        self._Q = Q

    def sample_action(self, obs, env = None):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        try:
            actions = env.state_actions(obs)
        except AttributeError:
            actions = self._Q.shape[1]

        return np.argmax(self._Q[obs, :actions])

    def __call__(self, obs):
        return self.sample_action(obs)


class SoftmaxPolicy():
    def __init__(self, Q):
        self._Q = Q

    def set_Q(self, Q):
        self._Q = Q

    def sample_action(self, obs, env = None):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        try:
            actions = env.state_actions(obs)
        except AttributeError:
            actions = self._Q.shape[1]
        exp = np.exp(self._Q[obs, :actions])
        probs = exp/np.sum(exp)
        return np.random.choice(np.arange(actions), p=probs)

    def __call__(self, obs):
        return self.sample_action(obs)

