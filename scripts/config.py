import data
import models
from aggregators import norm_lime
from samplers import el2n_score, entropy_sampler, max_variation_ratio, uniform_sampler

NAME_TO_MODEL_LOADER = {
    "emotion": models.emotion.load_model,
    "imdb": models.imdb.load_model,
}

NAME_TO_DATASET_LOADER = {
    "emotion": data.emotion.load_dataset,
    "imdb": data.imdb.load_dataset,
}

NAME_TO_SAMPLER = {
    "uniform": uniform_sampler.select_samples,
    "entropy": entropy_sampler.select_samples,
    "el2n": el2n_score.select_samples,
    "variation_ratio": max_variation_ratio.select_samples,
}

NAME_TO_AGGREGATOR = {"norm_lime": norm_lime.aggregate_local_explanations}
