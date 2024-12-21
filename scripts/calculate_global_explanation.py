# TODO: using selected aggregation method
# TODO: calculate ground truth (all samples) global explanation
# TODO: calculate compressed global explanation for some sample selection method
# TODO: compare both global explanations (MAE or TV)


import argparse
import json

import h5py
from config import NAME_TO_AGGREGATOR, NAME_TO_DATASET_LOADER, NAME_TO_MODEL_LOADER, NAME_TO_SAMPLER

from data.emotion import load_dataset
from models.emotion import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate global explanation")
    parser.add_argument("--dataset", type=str, help="Dataset to use", default="emotion")
    parser.add_argument("--explanation_file", type=str, help="Path to the explanation file")
    parser.add_argument("--num_samples", type=int, help="Number of samples to be selected")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--sampler", type=str, help="Sampler method", default="uniform")
    parser.add_argument(
        "--aggregator", type=str, help="Aggregator method", default="norm_lime"
    )

    return parser.parse_args()

def main(args):
    dataset_loader = NAME_TO_DATASET_LOADER[args.dataset]
    model_loader = NAME_TO_MODEL_LOADER[args.dataset]

    sampler = NAME_TO_SAMPLER[args.sampler]
    aggregator = NAME_TO_AGGREGATOR[args.aggregator]

    num_samples = args.num_samples
    with h5py.File(args.explanation_file, "r") as f:
        if num_samples == -1:
            num_samples = len(f)
        samples = sampler(f, num_samples, args.seed)
        # print(f"Selected samples: {samples}")

        global_explanation = aggregator(samples, f)
        _, class_indices = dataset_loader()
        _, tokenizer = model_loader()
        global_explanation = {
            class_indices[class_idx]: {
                tokenizer.decode([token_id]): token_score
                for token_id, token_score in token_scores.items()
            }
            for class_idx, token_scores in global_explanation.items()
        }
        json.dump(global_explanation, open(args.output_file, "w"))


if __name__ == "__main__":
    main(parse_args())