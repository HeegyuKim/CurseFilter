from attr import attr
from transformers_interpret import SequenceClassificationExplainer


class CurseFilter:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text, threshold=0.1, replace_token="*", return_explainer=False):
        cls_explainer = SequenceClassificationExplainer(self.model, self.tokenizer)

        attrs = cls_explainer(text)

        outputs = []

        for token, p in attrs:
            if token in ["[CLS]", "[SEP]"]:
                continue

            sep = not token.startswith("##")

            if p >= threshold:
                token = replace_token * len(token)

            if sep:
                outputs.append(token)
            else:
                outputs[-1] = outputs[-1] + token[2:]

        filtered_text = " ".join(outputs)
        if return_explainer:
            return filtered_text, cls_explainer
        else:
            return filtered_text
