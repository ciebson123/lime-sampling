# TODO: select indices of the samples to be used for calculating the global explanation
# TODO input: number of samples to be selected, **kwargs
# TODO: output: list of indices

import numpy as np
import h5py


def select_samples(file: h5py.File, num_samples: int, seed: int, **kwargs) -> list:
    # print(file.keys())
    np.random.seed(seed)
    return np.random.choice(list(file.keys()), num_samples, replace=False)
