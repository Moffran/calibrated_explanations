"""Utilities for managing random number generator state."""

import numpy as np


def set_rng_seed(seed: int) -> np.random.Generator:
    """Set the random number generator seed and return the generator.

    Parameters
    ----------
    seed : int
        The seed to be used in the random number generator.

    Returns
    -------
    np.random.Generator
        A new random number generator initialized with the given seed.
    """
    return np.random.default_rng(seed)


__all__ = ["set_rng_seed"]
