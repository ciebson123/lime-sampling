import datasets


def clip_num_samples(dataset: datasets.Dataset, max_samples: int = None):
    """Clip the number of samples in a dataset."""
    if max_samples is None:
        return dataset
    max_samples = min(len(dataset), max_samples)
    return dataset.select(range(max_samples))