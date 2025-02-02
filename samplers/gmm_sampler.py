import h5py
import numpy as np
from sklearn.mixture import GaussianMixture


def select_samples(file: h5py.File, num_samples: int, seed: int, num_classes: int) -> list:
    embeddings = []
    i_to_key = {}
    for i, (key, grp) in enumerate(file.items()):
        embeddings.append(grp["cls"][:])
        i_to_key[i] = key

    embeddings = np.stack(embeddings, axis=0)
    # fit a GMM in the embedding space
    gmm = GaussianMixture(n_components=num_classes, random_state=seed, covariance_type='full').fit(embeddings)
    # sample from fitted distribution
    samples, _ = gmm.sample(n_samples=2*num_samples)
    ids = []
    for sample in samples:
        # find the closest embeddings to the samples
        closest_idx = np.argmin(np.linalg.norm(embeddings - sample, axis=1)).item()
        closest_key = i_to_key[closest_idx]
        if closest_key not in ids:
            ids.append(closest_key)
        if len(ids) == num_samples:
            break
    print(f"Number of sampled indices: {len(ids)}")
    return ids
        

    


                