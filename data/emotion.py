import datasets

CATEGORIES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def download_and_save_dataset(path: str):
    data = datasets.load_dataset("dair-ai/emotion")
    data.save_to_disk(path)


def load_dataset(path: str):
    return datasets.load_from_disk(path)
