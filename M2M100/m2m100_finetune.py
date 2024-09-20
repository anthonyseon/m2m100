import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="M2M100 Translator")

# 타이틀
st.title("M2M100 Finetune Translation using Streamlit")

# 설명
st.write("Enter a sentence to translate between different languages using Meta's M2M100 model.")

# 모델 로드 함수
@st.cache_resource
def load_model():
    model_name = "JamesKim/m2m100-ft3"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# 입력 필드와 선택 상자
source_lang = st.selectbox("Select source language", ["en", "ko", "fr", "de", "ja", "zh"])
target_lang = st.selectbox("Select target language", ["en", "ko", "fr", "de", "ja", "zh"])
text_input = st.text_area("Input text to translate", height=150)

# 번역 버튼
if st.button("Translate"):
    if not text_input: 
        st.warning("Please enter text for translation!")
    else:
        tokenizer.src_lang = source_lang
        inputs = tokenizer(text_input, return_tensors="pt")

        with torch.no_grad():
            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))

        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        # 번역된 텍스트 출력
        st.subheader("Translated Text")
        st.write(translation)


