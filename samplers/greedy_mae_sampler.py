import numpy as np
import h5py
import json
from notebooks.utils import compute_mae, get_top_k_tokens_per_category
from copy import deepcopy
from collections import defaultdict

def incremental_norm_lime(scores, num_occurences_class_token, file: h5py.File, new_sample_idx: str):
    grp = file[new_sample_idx]
    predicted_label_idx = grp["predicted_label_idx"][()].item()
    token_ids = grp["token_ids"][:]
    token_scores = grp["token_scores"][:]
    total_sample_score = np.abs(token_scores).sum()
    for token_id, token_score in zip(token_ids, token_scores):
        scores[predicted_label_idx][token_id] *= num_occurences_class_token[predicted_label_idx][token_id] 
        scores[predicted_label_idx][token_id] += token_score ** 2 / total_sample_score
        num_occurences_class_token[predicted_label_idx][token_id] += 1
        scores[predicted_label_idx][token_id] /= num_occurences_class_token[predicted_label_idx][token_id]
        
    return scores, num_occurences_class_token

def un_incremental_norm_lime(scores, num_occurences_class_token, file: h5py.File, new_sample_idx: str):
    grp = file[new_sample_idx]
    predicted_label_idx = grp["predicted_label_idx"][()].item()
    token_ids = grp["token_ids"][:]
    token_scores = grp["token_scores"][:]
    total_sample_score = np.abs(token_scores).sum()
    for token_id, token_score in zip(token_ids, token_scores):
        scores[predicted_label_idx][token_id] *= num_occurences_class_token[predicted_label_idx][token_id] 
        scores[predicted_label_idx][token_id] -= token_score ** 2 / total_sample_score
        num_occurences_class_token[predicted_label_idx][token_id] -= 1
        if num_occurences_class_token[predicted_label_idx][token_id] != 0:
            scores[predicted_label_idx][token_id] /= num_occurences_class_token[predicted_label_idx][token_id]
        else:
            scores[predicted_label_idx][token_id] = 0
    return scores, num_occurences_class_token

def select_samples(
    file: h5py.File,
    num_samples: int,
    seed: int,
    ground_truth_results_path: str,
    top_k_gt: int,
    **kwargs
) -> list:

    gt_result = json.load(open(ground_truth_results_path, "r"))
    gt_result = get_top_k_tokens_per_category(gt_result, top_k_gt)
    used_ids_set = set()
    chosen_ids = []
    scores = defaultdict(lambda: defaultdict(float))
    num_occurences_class_token = defaultdict(lambda: defaultdict(int))
    for _ in range(num_samples):
        best_id = None
        best_mae = np.inf
        for key in file.keys():
            if key in used_ids_set:
                continue
            scores, num_occurences_class_token = incremental_norm_lime(
                    scores, num_occurences_class_token, file, key
                )
            mae = compute_mae(gt_result, scores, top_k_gt)
            if mae < best_mae:
                best_mae = mae
                best_id = key
                
            scores, num_occurences_class_token = un_incremental_norm_lime(
                    scores, num_occurences_class_token, file, key
                )
            
        chosen_ids.append(best_id)
        used_ids_set.add(best_id)
        scores, num_occurences_class_token = incremental_norm_lime(
            scores, num_occurences_class_token, file, best_id
        )

    return chosen_ids
