import datasets


def load_dataset(cache_dir: str = None):
    dataset = datasets.load_dataset("Yelp/yelp_review_full", split="train", cache_dir=cache_dir)
    class_names = dataset.features["label"].names
    return dataset, class_names
