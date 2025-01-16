import numpy as np
import h5py
from goodpoints import kt
from functools import partial


def gaussian_kernel(y, X, sigma=1.0):
    diff = y - X
    return np.exp(-np.sum(diff ** 2, axis=1) / (2 * sigma ** 2))


# to jest jakis log, ale no proszę się nie czepiac
def compute_m(n, num_samples):
    current_num_points = n
    m = 0
    while np.ceil(current_num_points / 2) > num_samples:
        m += 1
        current_num_points = np.ceil(current_num_points / 2)
    return m


def select_samples(file: h5py.File, num_samples: int, seed: int, **kwargs) -> list:
    cls_embeds = []

    for _, value in file.items():
        cls = value["cls"][:]
        cls_embeds.append(cls)

    cls_embeds = np.array(cls_embeds, dtype=np.float64)
    n = cls_embeds.shape[0]
    m = compute_m(n, num_samples)
    d = cls_embeds.shape[1]
    sigma = np.sqrt(2 * d)
    kernel = partial(gaussian_kernel, sigma=sigma)

    id_compressed = kt.thin(
        X=cls_embeds, m=m, split_kernel=kernel, swap_kernel=kernel, delta=0.5, seed=seed
    )

    print(
        f"kernel thinning takes {num_samples} out of recommended {len(id_compressed)}."
    )

    keys = list(file.keys())
    str_indices = [keys[idx] for idx in id_compressed[:num_samples]]

    return str_indices
