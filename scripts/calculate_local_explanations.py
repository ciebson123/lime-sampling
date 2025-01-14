import argparse
import os

import h5py
from tqdm import tqdm
from transformers import set_seed

from data.utils import clip_num_samples
from explainers.lime import LimeExplainer
from models.utils import predict
from scripts.config import NAME_TO_DATASET_LOADER, NAME_TO_MODEL_LOADER
from utils import setup_device

project_dir = os.environ["PROJECT_DIR"]
cache_dir = os.path.join(project_dir, ".cache")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate and save all local explanations."
    )
    parser.add_argument(
        "--dataset", type=str, default="emotion", help="Dataset to use."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments/emotion/lime",
        help="Directory to save the explanations.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for model predictions."
    )
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use.",
    )
    parser.add_argument(
        "--lime_num_features",
        type=int,
        default=None,
        help="Number of features to use in the explanation.",
    )
    parser.add_argument(
        "--lime_num_samples",
        type=int,
        default=5000,
        help="Number of samples to use in the explanation.",
    )
    parser.add_argument(
        "--lime_token_masking_strategy",
        type=str,
        default="remove",
        help="Token masking strategy.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def main(args):
    print(f"Arguments: {vars(args)}")
    set_seed(args.seed)
    device = setup_device()

    model_loader = NAME_TO_MODEL_LOADER[args.dataset]
    model, tokenizer = model_loader(cache_dir)
    model = model.eval()
    model = model.to(device)

    dataset_loader = NAME_TO_DATASET_LOADER[args.dataset]
    dataset, class_names = dataset_loader(cache_dir)
    dataset = clip_num_samples(dataset, args.max_num_samples)

    lime_args = {
        "batch_size": args.batch_size,
        "num_features": args.lime_num_features,
        "num_samples": args.lime_num_samples,
        "token_masking_strategy": args.lime_token_masking_strategy,
    }
    print(f"Lime arguments: {lime_args}")
    explainer = LimeExplainer(model, tokenizer, device, **lime_args)

    accuracy = 0

    experiment_dir = os.path.join(project_dir, args.experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    output_path = os.path.join(experiment_dir, "explanations.h5")
    if os.path.exists(output_path):
        print(f"Explanations already calculated, saved to {output_path}")
        exit()

    with h5py.File(output_path, "w") as f:
        for i, sample in tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc="Explaining model predictions...",
        ):
            text = sample["text"]

            probabilities = predict(model, tokenizer, text)
            predicted_label_idx = probabilities.argmax().item()
            accuracy += predicted_label_idx == sample["label"]

            explanation = explainer.explain(sample["text"])

            assert (
                predicted_label_idx == explanation.label
            ), "Predicted label and explanation label do not match"

            grp = f.create_group(str(i))
            grp.create_dataset("label_idx", data=sample["label"])
            grp.create_dataset("predicted_label_idx", data=predicted_label_idx)
            grp.create_dataset("probabilities", data=probabilities.squeeze().numpy())
            grp.create_dataset("token_ids", data=explanation.token_ids)
            grp.create_dataset("token_scores", data=explanation.token_scores)
            grp.create_dataset("explanation_fit", data=explanation.explanation_fit)
            grp.create_dataset("cls", data=explanation.cls)

    print(f"Accuracy: {accuracy / len(dataset)}")
    print(f"Explanations saved to {os.path.join(experiment_dir, 'explanations.h5')}")
    print("Done!")


if __name__ == "__main__":
    main(parse_args())
