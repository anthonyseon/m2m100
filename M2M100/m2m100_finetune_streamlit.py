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
