import math

import numpy as np
import torch
from tqdm import tqdm

from utils import dict_to_device


@torch.no_grad()
def predict(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = dict_to_device(inputs, model.device)
    outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=-1).cpu()


@torch.no_grad()
def predict_batched(
    model,
    tokenizer,
    texts: str | list[str],
    batch_size: int = 64,
    show_progress: bool = False,
    **tokenizer_kwargs,
):
    if not isinstance(texts, list):
        texts = [texts]

    n_batches = math.ceil(len(texts) / batch_size)
    batches = np.array_split(texts, n_batches)

    logits = []

    for batch in tqdm(
        batches,
        total=n_batches,
        leave=False,
        disable=not show_progress,
    ):
        inputs = tokenizer(batch.tolist(), padding=True, return_tensors="pt", **tokenizer_kwargs)
        inputs = dict_to_device(inputs, model.device)

        outputs = model(**inputs)
        logits.append(outputs.logits)

    logits = torch.cat(logits, dim=0)

    return torch.softmax(logits, dim=-1).cpu()