from lime.lime_text import LimeTextExplainer


def explain_model(samples, pipeline, categories, num_features=6):
    explainer = LimeTextExplainer(class_names=categories)

    explanations = [
        explainer.explain_instance(text, pipeline, num_features=num_features)
        for text in samples["text"]
    ]
    return explanations
