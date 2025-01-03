import argparse
import json
import os

import h5py
from config import (
    NAME_TO_AGGREGATOR,
    NAME_TO_DATASET_LOADER,
    NAME_TO_MODEL_LOADER,
    NAME_TO_SAMPLER,
)

project_dir = os.environ['PROJECT_DIR']
cache_dir = os.path.join(project_dir, ".cache")

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate global explanation")
    parser.add_argument("--dataset", type=str, help="Dataset to use", default="emotion")
    parser.add_argument("--explanation_file", type=str, help="Path to the explanation file", required=True)
    parser.add_argument("--num_samples", type=int, help="Number of samples to be selected")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--sampler", type=str, help="Sampler method", default="uniform")
    parser.add_argument(
        "--aggregator", type=str, help="Aggregator method", default="norm_lime"
    )

    return parser.parse_args()

def main(args):
    print(f"Arguments: {vars(args)}")
    dataset_loader = NAME_TO_DATASET_LOADER[args.dataset]
    model_loader = NAME_TO_MODEL_LOADER[args.dataset]

    sampler = NAME_TO_SAMPLER[args.sampler]
    aggregator = NAME_TO_AGGREGATOR[args.aggregator]

    num_samples = args.num_samples
    with h5py.File(args.explanation_file, "r") as f:
        if num_samples == -1:
            num_samples = len(f)
        
        print(f"Sampling {num_samples} local explanations using {args.sampler} method...")
        samples = sampler(f, num_samples, args.seed)

        print(f"Calculating global explanation using {args.aggregator} method...")
        global_explanation = aggregator(samples, f)

        _, class_names = dataset_loader(cache_dir)
        _, tokenizer = model_loader(cache_dir)
        global_explanation = {
            class_names[class_idx]: {
                tokenizer.decode([token_id]): token_score
                for token_id, token_score in token_scores.items()
            }
            for class_idx, token_scores in global_explanation.items()
        }
        json.dump(global_explanation, open(args.output_file, "w"))
        print(f"Global explanation saved to {args.output_file}")


if __name__ == "__main__":
    main(parse_args())