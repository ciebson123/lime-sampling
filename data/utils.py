import datasets


def clip_num_samples(dataset: datasets.Dataset, max_samples: int = None, seed: int = 42) -> datasets.Dataset:
    """Clip the number of samples in a dataset."""
    if max_samples is None:
        return dataset
    max_samples = min(len(dataset), max_samples)
    # shuffle before clipping
    return dataset.shuffle(seed=seed).select(range(max_samples))