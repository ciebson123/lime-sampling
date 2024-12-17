from samplers import max_variation_ratio, uniform_sampler, entropy_sampler

from aggregators import norm_lime


NAME_TO_SAMPLER = {
    "uniform": uniform_sampler.select_samples,
    "entropy": entropy_sampler.select_samples,
    "variation_ratio": max_variation_ratio.select_samples,
}

NAME_TO_AGGREGATOR = {"norm_lime": norm_lime.aggregate_local_explanations}
