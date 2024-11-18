import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline,
)
from functools import partial
import numpy as np


def download_and_save_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        "bhadresh-savani/distilbert-base-uncased-emotion"
    )
    tokenizer.save_pretrained(tokenizer_path)


def download_and_save_model(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        "bhadresh-savani/distilbert-base-uncased-emotion"
    )
    model.save_pretrained(model_path)


def download_and_save_pipeline(tokenizer_path: str, model_path: str):
    download_and_save_tokenizer(tokenizer_path)
    download_and_save_model(model_path)


def pipe_wrapper(pipe, text) -> Pipeline:
    scores = pipe(text, top_k=None)
    scorelist = [[d["score"] for d in score_example] for score_example in scores]
    return np.array(scorelist)


def load_pipeline(tokenizer_path: str, model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipe = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer, device=device
    )
    return partial(pipe_wrapper, pipe)
