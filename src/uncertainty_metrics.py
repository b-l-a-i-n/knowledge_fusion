import numpy as np


def maxprob(probs):
    """Compute the maximum probability."""
    return np.max(probs)


def margin(probs):
    """Compute the margin between the highest and the second-highest probability."""
    sorted_probs = np.sort(probs)[::-1]
    return sorted_probs[0] - sorted_probs[1]


def entropy(probs, eps=1e-12):
    """Compute entropy of the probability distribution."""
    return -np.sum(probs * np.log(probs + eps))
