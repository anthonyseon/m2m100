import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="M2M100 Translator")

# 모델을 로드하는 함수
@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def finetune_load_model():
    model_name = "JamesKim/m2m100-ft3"
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# 탭 구조
tab1, tab2 = st.tabs(["General (facebook/m2m100_418M)", "Finetune (JamesKim/m2m100-ft3_073)"])

# --- 일반 탭 ---
with tab1:
    st.title("M2M100 General Translation")

    # 설명
    st.write("Use Meta's M2M100 model to translate between different languages.")

    # 일반 모델 로드
    model_general, tokenizer_general = load_model()

    # 입력 필드
    source_lang_general = st.selectbox("Select source language", ["ko", "en", "fr", "de", "ja", "zh"], key="general_source")
    target_lang_general = st.selectbox("Select target language", ["en", "ko", "fr", "de", "ja", "zh"], key="general_target")
    text_input_general = st.text_area("Input text to translate", height=150, key="general_input")

    # 번역 버튼
    if st.button("Translate", key="general_translate"):
        if not text_input_general:
            st.warning("Please enter text for translation!")
        else:
            tokenizer_general.src_lang = source_lang_general
            inputs_general = tokenizer_general(text_input_general, return_tensors="pt")

            with torch.no_grad():
                generated_tokens_general = model_general.generate(**inputs_general, forced_bos_token_id=tokenizer_general.get_lang_id(target_lang_general))

            translation_general = tokenizer_general.batch_decode(generated_tokens_general, skip_special_tokens=True)[0]

            # 번역 결과 저장
            st.session_state.general_translation = translation_general

    # 번역 결과 표시
    if "general_translation" in st.session_state:
        st.subheader("Translated Text (General)")
        st.write(st.session_state.general_translation)

# --- 파인튜닝 탭 ---
with tab2:
    st.title("M2M100 Finetuned Translation")

    # 설명
    st.write("Use the fine-tuned M2M100 model from Hugging Face to translate.")

    # 파인튜닝된 모델 로드
    model_finetune, tokenizer_finetune = finetune_load_model()

    # 입력 필드
    source_lang_finetune = st.selectbox("Select source language", ["ko", "en", "fr", "de", "ja", "zh"], key="finetune_source")
    target_lang_finetune = st.selectbox("Select target language", ["en", "ko", "fr", "de", "ja", "zh"], key="finetune_target")
    text_input_finetune = st.text_area("Input text to translate", height=150, key="finetune_input")

    # 번역 버튼
    if st.button("Translate", key="finetune_translate"):
        if not text_input_finetune:
            st.warning("Please enter text for translation!")
        else:
            tokenizer_finetune.src_lang = source_lang_finetune
            inputs_finetune = tokenizer_finetune(text_input_finetune, return_tensors="pt")

            with torch.no_grad():
                generated_tokens_finetune = model_finetune.generate(**inputs_finetune, forced_bos_token_id=tokenizer_finetune.get_lang_id(target_lang_finetune))

            translation_finetune = tokenizer_finetune.batch_decode(generated_tokens_finetune, skip_special_tokens=True)[0]

            # 번역 결과 저장
            st.session_state.finetune_translation = translation_finetune

    # 번역 결과 표시
    if "finetune_translation" in st.session_state:
        st.subheader("Translated Text (Finetune)")
        st.write(st.session_state.finetune_translation)
