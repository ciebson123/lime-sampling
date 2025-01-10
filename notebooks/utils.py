import numpy as np


def get_top_k_tokens_per_category(analysis_results: dict, top_k=10):
    top_k_tokens = {}
    for category in analysis_results:
        tokens = analysis_results[category]
        tokens = [(t, v) for t, v in tokens.items()]
        tokens = sorted(tokens, key=lambda x: np.abs(x[1]), reverse=True)
        tokens = tokens[:top_k]
        top_k_tokens[category] = {t: v for t, v in tokens}
    return top_k_tokens


def compute_mae(gt_results, sampled_resutls, top_k=10):
    # computes MAE on top_k tokens per category
    # if a token from sampled is not in gt, it gets skipped in MAE computation
    mae_per_category = []
    gt_top_k = get_top_k_tokens_per_category(gt_results, top_k)
    for category in gt_top_k:
        mae = 0
        gt_tokens = gt_top_k[category]
        sampled_tokens = sampled_resutls.get(category, {})
        for token in gt_tokens:
            if token in sampled_tokens:
                mae += np.abs(gt_tokens[token] - sampled_tokens[token])
            else:
                mae += np.abs(gt_tokens[token])

        mae_per_category.append(mae / top_k)  # (top_k * len(gt_top_k))
    return np.mean(mae_per_category)


def recall_at_k(gt_results, sampled_results, top_k=10):
    # computes how many tokens from top_k in gt are in top_k in sampled
    recall_per_category = []
    gt_top_k = get_top_k_tokens_per_category(gt_results, top_k)
    sampled_top_k = get_top_k_tokens_per_category(sampled_results, top_k)
    for category in gt_top_k:
        recall = 0
        gt_tokens = gt_top_k[category]
        sampled_tokens = sampled_top_k.get(category, {})
        for token in gt_tokens:
            if token in sampled_tokens:
                recall += 1
        recall_per_category.append(recall / top_k)
    return np.mean(recall_per_category)
