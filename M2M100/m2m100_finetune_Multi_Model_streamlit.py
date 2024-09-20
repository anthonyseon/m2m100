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

def finetune_load_model(model_name):
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
    model_finetune, tokenizer_finetune = finetune_load_model("JamesKim/m2m100-ft3")

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
