from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(cache_dir: str = None):
    model_name_or_path = "bhadresh-savani/distilbert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    return model, tokenizer