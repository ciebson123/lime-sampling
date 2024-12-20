import numpy as np
import h5py


def select_samples(file: h5py.File, num_samples: int, seed: int, **kwargs) -> list:
    id_with_entropy = []

    for key, value in file.items():
        probabilities = value["probabilities"][:]
        entropy = np.sum(-probabilities * np.log2(probabilities))
        id_with_entropy.append((key, entropy))

    id_with_entropy = sorted(id_with_entropy, key=lambda x: x[1], reverse=True)
    return [x[0] for x in id_with_entropy[:num_samples]]
