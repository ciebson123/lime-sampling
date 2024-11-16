import numpy as np
from datasets import DatasetDict


def sample(dataset: DatasetDict, split: str, n_samples: int):
    # sample n_samples indices from the dataset from the specified split
    data_split = dataset[split]
    idx = np.random.choice(len(data_split), n_samples, replace=False)
    samples = data_split[idx]
    return samples
