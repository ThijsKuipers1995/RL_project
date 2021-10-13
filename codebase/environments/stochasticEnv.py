import numpy as np
from gym.envs.toy_text import discrete

rng = np.random.default_rng()

class stochasticEnv(discrete.DiscreteEnv):
    def step(self, a):
        transitions = self.P[self.s][a]
        i = rng.choice(len(transitions), 1, p=[t[0] for t in transitions])[0]
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r(), d, {"prob": p})
