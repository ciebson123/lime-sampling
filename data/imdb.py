import datasets


def load_dataset(cache_dir: str = None):
    dataset = datasets.load_dataset("stanfordnlp/imdb", split="test", cache_dir=cache_dir)
    class_names = dataset.features["label"].names
    return dataset, class_names
