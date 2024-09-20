'''
import streamlit as st

# 모델 및 데이터 관련 모듈을 불러옵니다.
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# M2M 클래스 선언 (이전 코드를 사용하여 정의된 클래스)
class M2M:
    def __init__(self, model_name:str, src_lang:str="ko", tgt_lang:str="en"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.tokenizer.src_lang = src_lang
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def trans(self, input_text:str):
        encoded_pt = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(**encoded_pt, 
            forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang))
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return output[0]

# 파인튜닝 된 모델을 사용하는 M2M 클래스를 추가로 정의
class FineTunedM2M(M2M):
    def __init__(self, model_dir:str, src_lang:str="ko", tgt_lang:str="en"):
        super().__init__(model_dir, src_lang, tgt_lang)

# 원본 모델과 파인튜닝 모델 초기화
m2m = M2M(model_name="facebook/m2m100_418M")
fine_tuned_m2m = FineTunedM2M(model_dir="./ft_fold")

# Streamlit 인터페이스 구현
st.title("Translation with Fine-tuning")

# 입력 텍스트 받기
input_text = st.text_area("Input Text", "안녕하세요")

# 'Translate' 버튼
if st.button('Translate'):
    translated_text = m2m.trans(input_text)
    st.text_area("Translated Text", translated_text)

# 'Fine-tune' 버튼
if st.button('Fine-tune'):
    fine_tuned_text = fine_tuned_m2m.trans(input_text)
    st.text_area("Fine-tuned Text", fine_tuned_text)
'''

import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="M2M100 Translator")

# Function to load a model
@st.cache_resource
def load_model(model_name):
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Tab structure
tab1, tab2 = st.tabs(["General", "Finetune"])

# --- General tab ---
with tab1:
    st.title("M2M100 General Translation")

    # Description
    st.write("Enter a sentence to translate between different languages using Meta's M2M100 model.")

    # Load the general model
    model_general, tokenizer_general = load_model("facebook/m2m100_418M")

    # Input fields
    source_lang_general = st.selectbox("Select source language", ["en", "ko", "fr", "de", "ja", "zh"], key="general_source")
    target_lang_general = st.selectbox("Select target language", ["en", "ko", "fr", "de", "ja", "zh"], key="general_target")
    text_input_general = st.text_area("Input text to translate", height=150, key="general_input")

    # Translation button
    if st.button("Translate", key="general_translate"):
        if not text_input_general:
            st.warning("Please enter text for translation!")
        else:
            tokenizer_general.src_lang = source_lang_general
            inputs_general = tokenizer_general(text_input_general, return_tensors="pt")

            with torch.no_grad():
                generated_tokens_general = model_general.generate(**inputs_general, forced_bos_token_id=tokenizer_general.get_lang_id(target_lang_general))

            translation_general = tokenizer_general.batch_decode(generated_tokens_general, skip_special_tokens=True)[0]

            # Display the translation
            st.subheader("Translated Text (General)")
            st.write(translation_general)

# --- Finetune tab ---
with tab2:
    st.title("M2M100 Finetuned Translation")

    # Description
    st.write("Enter a sentence to translate using the fine-tuned M2M100 model from Hugging Face.")

    # Load the fine-tuned model
    model_finetune, tokenizer_finetune = load_model("JamesKim/m2m100-ft3")

    # Input fields
    source_lang_finetune = st.selectbox("Select source language", ["en", "ko", "fr", "de", "ja", "zh"], key="finetune_source")
    target_lang_finetune = st.selectbox("Select target language", ["en", "ko", "fr", "de", "ja", "zh"], key="finetune_target")
    text_input_finetune = st.text_area("Input text to translate", height=150, key="finetune_input")

    # Translation button
    if st.button("Translate", key="finetune_translate"):
        if not text_input_finetune:
            st.warning("Please enter text for translation!")
        else:
            tokenizer_finetune.src_lang = source_lang_finetune
            inputs_finetune = tokenizer_finetune(text_input_finetune, return_tensors="pt")

            with torch.no_grad():
                generated_tokens_finetune = model_finetune.generate(**inputs_finetune, forced_bos_token_id=tokenizer_finetune.get_lang_id(target_lang_finetune))

            translation_finetune = tokenizer_finetune.batch_decode(generated_tokens_finetune, skip_special_tokens=True)[0]

            # Display the translation
            st.subheader("Translated Text (Finetune)")
            st.write(translation_finetune)
