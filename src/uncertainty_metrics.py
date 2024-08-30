import numpy as np
from scipy.special import softmax
import heapq


def perplexity(probs):
    """Compute perplexity, a measure of uncertainty, using the probability distribution."""
    return np.exp(-np.dot(probs, np.log(probs)))


def maxprob(probs):
    """Compute the maximum probability."""
    return max(probs)


def margin(probs):
    """Compute the margin between the highest and the second highest probability."""
    sorted_probs = sorted(probs, reverse=True)
    return sorted_probs[0] - sorted_probs[1]


def entropy(probs, eps=1e-12):
    """Compute entropy of the probability distribution."""
    return -np.sum(probs * np.log(probs + eps))
