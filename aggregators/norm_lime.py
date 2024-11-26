# TODO: aggregate local explanations from .h5 file into a global explanation
# TODO: using the normLIME method from GLIME paper (https://www.sciencedirect.com/science/article/abs/pii/S0004370222001631)
# TODO: input: list of sample indices, .h5 file with local explanations
# TODO: output: class -> token_ids -> token_scores for the global explanation

# from dataclasses import dataclass
import h5py
import numpy as np
from collections import defaultdict


def aggregate_local_explanations(
    sample_indices: list[str], file: h5py.File
) -> dict[int, dict[int, float]]:
    global_explanation = defaultdict(lambda: defaultdict(float))
    num_occurences_class_token = defaultdict(lambda: defaultdict(int))
    for sample_idx in sample_indices:
        grp = file[sample_idx]
        predicted_label_idx = grp["predicted_label_idx"][()].item()
        token_ids = grp["token_ids"][:]
        token_scores = grp["token_scores"][:]
        total_sample_score = np.abs(token_scores).sum().item()

        for token_id, token_score in zip(token_ids, token_scores):
            global_explanation[predicted_label_idx][token_id.item()] += (
                token_score.item() ** 2 / total_sample_score
            )
            num_occurences_class_token[predicted_label_idx][token_id.item()] += 1

    for class_idx in global_explanation.keys():
        for token_id in global_explanation[class_idx].keys():
            global_explanation[class_idx][token_id] /= num_occurences_class_token[
                class_idx
            ][token_id]

    return global_explanation
