import h5py
import numpy as np


def select_samples(file: h5py.File, num_samples: int, seed: int, **kwargs) -> list:
    np.random.seed(seed)
    return np.random.choice(list(file.keys()), num_samples, replace=False)
