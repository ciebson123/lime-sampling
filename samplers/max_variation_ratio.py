import h5py
import numpy as np


def select_samples(file: h5py.File, num_samples: int, seed: int, **kwargs) -> list:
    # https://arxiv.org/pdf/1703.02910 maximize 1 - max(p(y|x))
    id_with_variation_ratio = []

    for key, value in file.items():
        probabilities = value["probabilities"][:]
        variation_ratio = 1 - np.max(probabilities)
        id_with_variation_ratio.append((key, variation_ratio))

    id_with_variation_ratio = sorted(id_with_variation_ratio, key=lambda x: x[1], reverse=True)
    return [x[0] for x in id_with_variation_ratio[:num_samples]]
