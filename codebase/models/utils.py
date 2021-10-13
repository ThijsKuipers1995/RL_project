import numpy as np
from tqdm import tqdm as _tqdm


def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)


def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n