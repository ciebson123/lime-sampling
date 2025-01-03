import h5py
import numpy as np


def select_samples(file: h5py.File, num_samples: int, seed: int, **kwargs) -> list:
    # EL2N = Error L2 Norm from https://arxiv.org/pdf/2107.07075, https://openreview.net/pdf?id=1dwXa9vmOI
    # given as ||p(y | x) - y||_2
    id_with_el2n_score = []

    for key, value in file.items():
        probabilities = value["probabilities"][:]
        label_idx = value["label_idx"][()]
        y = np.zeros(probabilities.shape)
        y[label_idx] = 1
        el2n_score = np.linalg.norm(probabilities - y, ord=2)
        id_with_el2n_score.append((key, el2n_score))

    id_with_el2n_score = sorted(id_with_el2n_score, key=lambda x: x[1], reverse=True)
    return [x[0] for x in id_with_el2n_score[:num_samples]]
