from data.emotion import CATEGORIES, load_dataset
from models.emotion import load_pipeline
from samplers.uniform_sampler import sample
from explanations.lime import explain_model

dataset = load_dataset("storage/emotion_dataset")
pipeline = load_pipeline("storage/emotion_tokenizer", "storage/emotion_model", "cuda")

samples = sample(dataset, "validation", 10)
print(explain_model(samples, pipeline, CATEGORIES))
