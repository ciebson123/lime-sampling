import os

import h5py
from tqdm import tqdm
from transformers import set_seed

from data.emotion import load_dataset
from data.utils import clip_num_samples
from explanations.lime import LimeExplainer
from models.emotion import load_model
from models.utils import predict
from utils import setup_device

project_dir = os.environ['PROJECT_DIR']
cache_dir = os.path.join(project_dir, ".cache")
experiment_dir = os.path.join(project_dir, "expriments/emotion/lime")

seed = 42
max_num_samples = None
lime_args = {
    "num_features": None,
    "num_samples": 5000,
    "batch_size": 512,
    "token_masking_strategy": "remove",
}

if __name__ == "__main__":
    set_seed(seed)
    device = setup_device()

    model, tokenizer = load_model(cache_dir)
    model = model.eval()
    model = model.to(device)

    dataset, class_names = load_dataset(cache_dir)
    dataset = clip_num_samples(dataset, max_num_samples)

    explainer = LimeExplainer(model, tokenizer, device, **lime_args)

    accuracy = 0

    os.makedirs(experiment_dir, exist_ok=True)
    output_path = os.path.join(experiment_dir, "explanations.h5")
    if os.path.exists(output_path):
        print(f"Explanations already calculated, saved to {output_path}")
        exit()
    
    with h5py.File(output_path, "w") as f:
        for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Explaining model predictions..."):
            text = sample["text"]

            probabilities = predict(model, tokenizer, text)
            predicted_label_idx = probabilities.argmax().item()
            accuracy += predicted_label_idx == sample["label"]

            explanation = explainer.explain(sample["text"])

            assert predicted_label_idx == explanation.label, "Predicted label and explanation label do not match"

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
