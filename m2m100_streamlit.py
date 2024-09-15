import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Streamlit 페이지 설정
st.set_page_config(layout="centered", page_title="M2M100 Translator")

# 타이틀
st.title("M2M100 Translation using Streamlit")

# 설명
st.write("""
Translate text using Meta's M2M100 model. 
Choose the source language and target language, input your text, and get the translation.
""")

# 모델과 토크나이저 로드
@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# 입력을 위한 Streamlit 위젯
source_lang = st.selectbox("Select source language", ["ko", "en", "fr", "de", "ja", "zh"])
target_lang = st.selectbox("Select target language", ["ko", "en", "fr", "de", "ja", "zh"])
text_input = st.text_area("Input text to translate", height=150)

# 번역 버튼
if st.button("Translate"):
    if not text_input:
        st.warning("Please enter text for translation!")
    else:
        # 입력 텍스트 토크나이징 및 번역
        tokenizer.src_lang = source_lang
        inputs = tokenizer(text_input, return_tensors="pt")
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        
        # 번역된 텍스트 디코딩
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # 결과 출력
        st.subheader("Translated Text")
        st.write(translation)