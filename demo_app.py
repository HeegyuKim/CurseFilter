import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer
import pandas as pd
import os

from src.lit_module import TextClassificationStudentModule
from src.curse import CurseFilter

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = "cuda" if torch.cuda.is_available() else "cpu"
st.title("Classifier demo app")


@st.cache(allow_output_mutation=True)
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "monologg/koelectra-small-v3-discriminator"
    )
    return tokenizer


@st.cache()
def get_modules():
    hate = "trained_models/hate.ckpt"
    curse = "trained_models/curse.ckpt"
    curse = TextClassificationStudentModule.load_from_checkpoint(curse, device)
    hate = TextClassificationStudentModule.load_from_checkpoint(hate, device)
    return curse, hate


with st.spinner("모델을 가져오는 중..."):
    tokenizer = get_tokenizer()
    curse_model, hate_model = get_modules()
    curse_filter = CurseFilter(curse_model.model, tokenizer)


def display_explanation(model, tokenizer, text):
    ex = SequenceClassificationExplainer(model, tokenizer)
    ex(text)
    components.html(ex.visualize().data)


text = "마춤법 틀릴수도 잇지 게세끼야"
text = st.text_area("혐오표현이 들어간 문장을 입력해보세요", text)

with st.spinner("문장을 분석중이에요..."):
    with torch.no_grad():
        model_input = tokenizer(text, return_tensors="pt")
        curse_pred = curse_model(
            model_input["input_ids"], model_input["attention_mask"]
        )[0].item()
        hate_pred = hate_model(model_input["input_ids"], model_input["attention_mask"])[
            0
        ].item()

        result_df = pd.DataFrame({"혐오": [hate_pred], "욕설": [curse_pred]})
        st.dataframe(result_df)

        if curse_pred > 0.5:
            st.error("욕설 필터링 결과: " + curse_filter(text))

        st.header("Curse Explanation")
        display_explanation(curse_model.model, tokenizer, text)
        st.header("Hate Explanation")
        display_explanation(hate_model.model, tokenizer, text)
