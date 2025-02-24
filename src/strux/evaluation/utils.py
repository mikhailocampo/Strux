import numpy as np


def compute_entropy(p: float) -> float:
    """Compute binary entropy of p."""
    p = max(min(p, 1.0 - 1e-9), 1e-9)  # clamp to avoid log(0)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def probability_from_two_logprobs(logp1: float, logp2: float) -> float:
    """
    Compute normalized probability from two log probabilities.

    Given two log probabilities, logp1 and logp2, returns:
      p = exp(logp1) / (exp(logp1) + exp(logp2))
    """
    return np.exp(logp1) / (np.exp(logp1) + np.exp(logp2))