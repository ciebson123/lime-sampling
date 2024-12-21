from collections import namedtuple

import numpy as np
from lime.lime_text import LimeTextExplainer

from models.utils import predict_batched

ExplainationOutput = namedtuple(
    "ExplainationOutput", ["tokens", "token_ids", "token_scores", "label", "explanation_fit"]
)


class LimeExplainer:
    def __init__(
        self,
        model,
        tokenizer,
        num_features: int = None,
        num_samples: int = 1000,
        batch_size: int = 64,
        token_masking_strategy: str = "remove",
        mask_string: str = "UNKWORDZ",
    ):
        self.model = model
        self.tokenizer = tokenizer

        self._explainer = LimeTextExplainer(bow=False)
        self.num_features = num_features
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.token_masking_strategy = token_masking_strategy
        self.mask_string = mask_string

    def _forward_fn_token_ids(self, token_ids_texts: list[str]):
        if self.token_masking_strategy == "mask":
            mask_token_id = str(self.tokenizer.mask_token_id)
        elif self.token_masking_strategy == "pad":
            mask_token_id = str(self.tokenizer.pad_token_id)
        elif self.token_masking_strategy == "remove":
            mask_token_id = ""
        else:
            raise NotImplementedError(f"Token masking strategy {self.token_masking_strategy} not implemented")

        def process_token_ids(text: str):
            # fix replacing [CLS] with mask_string
            if text.startswith(self.mask_string):
                text = f"{self.tokenizer.cls_token_id} {text[len(self.mask_string)+1:]}"
            # fix replacing [SEP] with mask_string
            if text.endswith(self.mask_string):
                text = f"{text[:-len(self.mask_string)-1]} {self.tokenizer.sep_token_id}"
            # apply mask
            return text.replace(self.mask_string, mask_token_id)

        # apply masking strategy
        # and convert list of strings to list of ints
        token_ids = [[int(i) for i in process_token_ids(text).split(" ") if i != ""] for text in token_ids_texts]

        # back to list of strings
        masked_texts = self.tokenizer.batch_decode(token_ids)

        return predict_batched(
            self.model,
            self.tokenizer,
            masked_texts,
            add_special_tokens=False,  # already in the text
            batch_size=self.batch_size,
            show_progress=False,
        ).numpy()

    def explain(self, text: str) -> ExplainationOutput:
        encoded_text = self.tokenizer(text, return_tensors="pt", return_special_tokens_mask=True, max_length=512, truncation=True)
        special_tokens_mask = encoded_text.pop("special_tokens_mask")
        token_ids = encoded_text["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        explanation = self._explainer.explain_instance(
            " ".join([str(i) for i in token_ids]),
            self._forward_fn_token_ids,
            top_labels=1,
            num_samples=self.num_samples,
            num_features=self.num_features or len(token_ids),
        )

        token_scores = dict(sorted(explanation.local_exp[explanation.top_labels[0]]))
        # fill missing tokens with 0.0
        token_scores = np.array([token_scores.get(i, 0.0) for i in range(len(token_ids))])

        # remove special tokens from the explanation
        token_scores[special_tokens_mask.squeeze().bool().cpu().numpy()] = 0.0

        return ExplainationOutput(
            tokens,
            token_ids,
            token_scores,
            explanation.top_labels[0],
            explanation.score,
        )
