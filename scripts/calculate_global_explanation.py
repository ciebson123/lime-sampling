# TODO: using selected aggregation method
# TODO: calculate ground truth (all samples) global explanation
# TODO: calculate compressed global explanation for some sample selection method
# TODO: compare both global explanations (MAE or TV)


import h5py
import argparse
import json
from data.emotion import load_dataset
from models.emotion import load_model
from config import NAME_TO_AGGREGATOR, NAME_TO_SAMPLER

# add parser for explanation file
parser = argparse.ArgumentParser(description="Calculate global explanation")
parser.add_argument("--explanation_file", type=str, help="Path to the explanation file")
parser.add_argument("--num_samples", type=int, help="Number of samples to be selected")
parser.add_argument("--output_file", type=str, help="Path to the output file")
parser.add_argument("--seed", type=int, help="Random seed", default=42)
parser.add_argument("--sampler", type=str, help="Sampler method", default="uniform")
parser.add_argument(
    "--aggregator", type=str, help="Aggregator method", default="norm_lime"
)

args = parser.parse_args()
explanation_file = args.explanation_file
num_samples = args.num_samples
output_file = args.output_file
seed = args.seed
sampler = NAME_TO_SAMPLER[args.sampler]
aggregator = NAME_TO_AGGREGATOR[args.aggregator]

with h5py.File(explanation_file, "r") as f:
    if num_samples == -1:
        num_samples = len(f)
    samples = sampler(f, num_samples, seed)
    # print(f"Selected samples: {samples}")

    global_explanation = aggregator(samples, f)
    _, class_indices = load_dataset()
    _, tokenizer = load_model()
    global_explanation = {
        class_indices[class_idx]: {
            tokenizer.decode([token_id]): token_score
            for token_id, token_score in token_scores.items()
        }
        for class_idx, token_scores in global_explanation.items()
    }
    json.dump(global_explanation, open(output_file, "w"))
