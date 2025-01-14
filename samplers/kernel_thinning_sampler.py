import numpy as np
import h5py
from goodpoints import compress


def select_samples(file: h5py.File, num_samples: int, seed: int, **kwargs) -> list:
    cls_embeds = []

    for key, value in file.items():
        cls = value["probabilities"][:]
        cls_embeds.append(cls)

    cls_embeds = np.array(cls_embeds)
    n = cls_embeds.shape[0]
    d = cls_embeds.shape[1]
    sigma = np.sqrt(2 * d)

    id_compressed = compress.compresspp_kt(
        cls_embeds,
        kernel_type=b"gaussian",
        k_params=np.array([sigma ** 2]),
        g=4,
        seed=seed,
    )

    print(
        f"kernel thinning takes {num_samples} out of recommended {len(id_compressed)}."
    )

    return id_compressed[:num_samples]
