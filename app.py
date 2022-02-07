import streamlit as st
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
import streamlit.components.v1 as components


model_name = "beomi/kcbert-base"
pth_path = "trained_models/curse-classifier-kcbert.pth"


@st.cache()
def get_model():
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_config(config).eval()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    return model

def filter_curse(attrs, threshold=0.1):
    outputs = []

    for token, p in attrs:
        if token in ["[CLS]", "[SEP]"]:
            continue

        sep = not token.startswith("##")

        if p >= threshold:
            token = "*" * len(token)

        if sep:
            outputs.append(token)
        else:
            outputs[-1] = outputs[-1] + token[2:]
        
    return " ".join(outputs)

st.title("Korean Curse Filtering")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = get_model()


text = "마춤법 틀릴수도 잇지 게세끼야"
text = st.text_area('혐오표현이 들어간 문장을 입력해보세요', text)

with st.spinner('문장을 분석중이에요...'):
    model_input = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    pred = model(**model_input).logits.detach()[0]
    pred = torch.sigmoid(pred).numpy()[0]
    cls_explainer = SequenceClassificationExplainer(
      model,
      tokenizer)
    attrs = cls_explainer(text)

print(pred)

if pred > 0.5:
    st.error("필터링 결과: " + filter_curse(attrs))
else:
    st.success("혐오 표현이 없습니다 :)")
        
components.html(cls_explainer.visualize().data)
st.write(attrs)