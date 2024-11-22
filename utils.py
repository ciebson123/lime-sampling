import datasets
import numpy as np
import torch
from IPython.core.display import HTML, display


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def dict_to_device(d: dict, device: torch.device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def display_token_scores(tokens: list[str], token_scores: np.ndarray):
    rgb = lambda x: "255,0,0" if x < 0 else "0,255,0"
    alpha = lambda x: abs(x) ** 0.5
    token_marks = [
        f'<mark style="background-color:rgba({rgb(score)},{alpha(score)}); color:black;">{token}</mark>'
        for token, score in zip(tokens, token_scores.tolist())
    ]

    display(HTML('<p style="background-color:white;">' + " ".join(token_marks) + "</p>"))
