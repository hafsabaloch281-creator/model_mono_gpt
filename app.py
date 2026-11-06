import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. App Title and Layout
st.title("🌟 Sentiment Analysis with Decoder-Only Transformer")
st.write("Type a sentence below to analyze its sentiment using your fine-tuned model!")

# 2. Load Model and Tokenizer from your Hugging Face Account
@st.cache_resource
def load_model_and_tokenizer():
    # Replace with your exact Hugging Face model repository path
    model_id = "Ahm-d18/my-gpt2-sentiment" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model

with st.spinner("Loading your model from Hugging Face... Please wait! ⏳"):
    tokenizer, model = load_model_and_tokenizer()

# 3. User Input Box
user_input = st.text_area("Enter your text here:", placeholder="Type something like 'I love this product!'...")

# 4. Prediction Button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            # Tokenize input and generate response/classification
            inputs = tokenizer(user_input, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Display Result
            st.success("Analysis Complete!")
            st.write("**Output:**")
            st.code(prediction)
