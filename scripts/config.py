from aggregators import averaged_importance, norm_lime
from data import emotion as emotion_data
from data import imdb as imdb_data
from models import emotion as emotion_models
from models import imdb as imdb_models
from samplers import el2n_score, entropy_sampler, max_variation_ratio, uniform_sampler, gmm_sampler

NAME_TO_MODEL_LOADER = {
    "emotion": emotion_models.load_model,
    "imdb": imdb_models.load_model,
}

NAME_TO_DATASET_LOADER = {
    "emotion": emotion_data.load_dataset,
    "imdb": imdb_data.load_dataset,
}

NAME_TO_SAMPLER = {
    "uniform": uniform_sampler.select_samples,
    "entropy": entropy_sampler.select_samples,
    "el2n": el2n_score.select_samples,
    "variation_ratio": max_variation_ratio.select_samples,
    "gmm": gmm_sampler.select_samples,
}

NAME_TO_AGGREGATOR = {"norm_lime": norm_lime.aggregate_local_explanations}
