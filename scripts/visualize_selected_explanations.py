from collections import defaultdict
import argparse
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap

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


def visualiaze_latent_space(embeddings: np.ndarray, labels: np.ndarray = None):
    """Visualize latent space using UMAP."""
    projector = umap.UMAP()
    projected_embeddings = projector.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        x=projected_embeddings[:, 0],
        y=projected_embeddings[:, 1],
        hue=labels,
        ax=ax,
    )
    plt.axis("off")
    plt.legend(title="")

    return fig


def array_list(array_like) -> np.ndarray:
    return np.array(list(array_like))


def main(args):
    print(f"Arguments: {vars(args)}")
    dataset_loader = NAME_TO_DATASET_LOADER[args.dataset]
    _, class_names = dataset_loader(cache_dir)
    
    model_loader = NAME_TO_MODEL_LOADER[args.dataset]
    _, tokenizer = model_loader(cache_dir)
    vocab_size = tokenizer.vocab_size

    sampler = NAME_TO_SAMPLER[args.sampler]
    aggregator = NAME_TO_AGGREGATOR[args.aggregator]

    num_samples = args.num_samples
    with h5py.File(args.explanation_file, "r") as f:
        
        print(f"Calculating global explanation using {args.aggregator} method...")
        all_indices = sampler(f, len(f), args.seed)
        global_explanation_dict = aggregator(all_indices, f)
        global_explanations = defaultdict(lambda: np.zeros(vocab_size))
        for class_idx, token_scores in global_explanation_dict.items():
            token_ids = array_list(token_scores.keys())
            token_scores = array_list(token_scores.values())
            global_explanation = np.zeros(vocab_size)
            global_explanation[token_ids] = token_scores
            global_explanations[class_idx] = global_explanation

        print(f"Sampling {num_samples} local explanations using {args.sampler} method...")
        sample_indices = sampler(f, num_samples, args.seed)

        local_explanations = defaultdict(list)
        for sample_idx in sample_indices:
            grp = f[sample_idx]
            class_idx = grp["predicted_label_idx"][()].item()
            token_ids = grp["token_ids"][:]
            token_scores = np.abs(grp["token_scores"][:]) # global explanation contains absolute values so we do the same here
            # map into vocab space
            local_explanation = np.zeros(vocab_size)
            local_explanation[token_ids] = token_scores
            local_explanations[class_idx].append(local_explanation)

        local_explanations = {
            class_idx: np.array(local_explanations[class_idx])
            for class_idx in local_explanations
        }

    print("Visualizing latent space of global explanations...")
    for class_idx in global_explanations.keys():
        global_explanation = global_explanations[class_idx].reshape(1, -1)
        print(global_explanation.shape)
        local_explanation = local_explanations[class_idx]
        print(local_explanation.shape)
        embeddings = np.concatenate([local_explanation, global_explanation])
        labels = np.array(['local'] * len(local_explanation) + ['global'])
        fig = visualiaze_latent_space(embeddings, labels)
        plt.title(f"dataset = {args.dataset}, sampler = {args.sampler}, class = {class_names[class_idx]}")
        fig.savefig(f"{args.output_file}_{class_names[class_idx]}.png")
        plt.close(fig)

    print(f"Visualization saved to {args.output_file}")


if __name__ == "__main__":
    main(parse_args())