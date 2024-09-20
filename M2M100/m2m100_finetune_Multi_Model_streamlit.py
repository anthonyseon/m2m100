import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="M2M100 Translator")

# 일반 모델을 '/model/' 경로에서 불러오는 함수
@st.cache_resource
def load_model():
    # '/model/' 경로에서 모델과 토크나이저를 불러옵니다.
    model = M2M100ForConditionalGeneration.from_pretrained("/model/")
    tokenizer = M2M100Tokenizer.from_pretrained("/model/")
    return model, tokenizer

# 파인튜닝된 모델을 '/finetunemodel/' 경로에서 불러오는 함수
def finetune_load_model():
    # '/finetunemodel/' 경로에서 파인튜닝된 모델과 토크나이저를 불러옵니다.
    model = M2M100ForConditionalGeneration.from_pretrained("/finetunemodel/")
    tokenizer = M2M100Tokenizer.from_pretrained("/finetunemodel/")
    return model, tokenizer

# 탭 구조 설정
tab1, tab2 = st.tabs(["일반 번역", "파인튜닝 번역"])

# --- 일반 번역 탭 ---
with tab1:
    st.title("M2M100 일반 번역")

    # 설명
    st.write("Meta의 M2M100 모델을 사용하여 다양한 언어 간 번역을 수행합니다.")

    # 일반 모델 불러오기
    model_general, tokenizer_general = load_model()

    # 입력 필드 설정
    source_lang_general = st.selectbox("원본 언어 선택", ["en", "ko", "fr", "de", "ja", "zh"], key="general_source")
    target_lang_general = st.selectbox("목표 언어 선택", ["en", "ko", "fr", "de", "ja", "zh"], key="general_target")
    text_input_general = st.text_area("번역할 텍스트 입력", height=150, key="general_input")

    # 번역 버튼
    if st.button("번역하기", key="general_translate"):
        if not text_input_general:
            st.warning("번역할 텍스트를 입력해 주세요!")
        else:
            # 번역 수행
            tokenizer_general.src_lang = source_lang_general
            inputs_general = tokenizer_general(text_input_general, return_tensors="pt")

            with torch.no_grad():
                generated_tokens_general = model_general.generate(
                    **inputs_general, 
                    forced_bos_token_id=tokenizer_general.get_lang_id(target_lang_general)
                )

            translation_general = tokenizer_general.batch_decode(generated_tokens_general, skip_special_tokens=True)[0]

            # 번역 결과 출력
            st.subheader("번역 결과 (일반)")
            st.write(translation_general)

# --- 파인튜닝 번역 탭 ---
with tab2:
    st.title("M2M100 파인튜닝 번역")

    # 설명
    st.write("Hugging Face에서 제공하는 파인튜닝된 M2M100 모델을 사용한 번역입니다.")

    # 파인튜닝된 모델 불러오기
    model_finetune, tokenizer_finetune = finetune_load_model()

    # 입력 필드 설정
    source_lang_finetune = st.selectbox("원본 언어 선택", ["en", "ko", "fr", "de", "ja", "zh"], key="finetune_source")
    target_lang_finetune = st.selectbox("목표 언어 선택", ["en", "ko", "fr", "de", "ja", "zh"], key="finetune_target")
    text_input_finetune = st.text_area("번역할 텍스트 입력", height=150, key="finetune_input")

    # 번역 버튼
    if st.button("번역하기", key="finetune_translate"):
        if not text_input_finetune:
            st.warning("번역할 텍스트를 입력해 주세요!")
        else:
            # 파인튜닝된 모델을 사용한 번역 수행
            tokenizer_finetune.src_lang = source_lang_finetune
            inputs_finetune = tokenizer_finetune(text_input_finetune, return_tensors="pt")

            with torch.no_grad():
                generated_tokens_finetune = model_finetune.generate(
                    **inputs_finetune, 
                    forced_bos_token_id=tokenizer_finetune.get_lang_id(target_lang_finetune)
                )

            translation_finetune = tokenizer_finetune.batch_decode(generated_tokens_finetune, skip_special_tokens=True)[0]

            # 번역 결과 출력
            st.subheader("번역 결과 (파인튜닝)")
            st.write(translation_finetune)
