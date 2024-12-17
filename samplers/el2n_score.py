import numpy as np
import h5py


def select_samples(file: h5py.File, num_samples: int, seed: int, **kwargs) -> list:
    # https://arxiv.org/pdf/2107.07075 max p(y | x) - y
    id_with_el2n_score = []

    for key, value in file.items():
        probabilities = value["probabilities"][:]
        label_idx = value["label_idx"][()]
        el2n_score = abs(probabilities[label_idx] - 1)
        id_with_el2n_score.append((key, el2n_score))

    id_with_el2n_score = sorted(id_with_el2n_score, key=lambda x: x[1], reverse=True)
    return [x[0] for x in id_with_el2n_score[:num_samples]]
