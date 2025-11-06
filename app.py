import streamlit as st
import google.generativeai as genai
import os

# Configure Google Gemini API
# Ensure your API key is set as an environment variable or in Streamlit secrets
# For local development, you can set it as an environment variable:
# export GOOGLE_API_KEY="YOUR_API_KEY"
# For Streamlit Cloud, use st.secrets["GOOGLE_API_KEY"]
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Failed to configure Google Gemini API. Make sure GOOGLE_API_KEY is set. Error: {e}")
    st.stop()

st.set_page_config(page_title="Prompt Engineering Playground", layout="wide")

# Initialize the Generative Model
model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

st.title("Prompt Engineering Playground with Gemini")

# Sidebar for LLM parameters and prompt techniques
st.sidebar.header("LLM Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
top_k = st.sidebar.slider("Top K", min_value=0, max_value=100, value=40, step=1)

st.sidebar.header("Prompt Engineering Techniques")
prompt_technique = st.sidebar.selectbox(
    "Select a technique",
    ["None", "Few-shot", "Chain-of-Thought", "Persona-based"]
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to enrich prompt based on selected technique
def enrich_prompt(user_prompt, technique):
    if technique == "Few-shot":
        # Example few-shot: provide examples to guide the model
        return (
            "Here are a few examples:\n"
            "Q: What is the capital of France?\nA: Paris.\n"
            "Q: What is the capital of Japan?\nA: Tokyo.\n"
            f"Now, answer the following question based on the examples: {user_prompt}"
        )
    elif technique == "Chain-of-Thought":
        # Example Chain-of-Thought: encourage step-by-step reasoning
        return (
            "Let's think step by step. "
            f"When answering the following question, explain your reasoning: {user_prompt}"
        )
    elif technique == "Persona-based":
        # Example Persona-based: assign a persona to the model
        return (
            "You are a helpful and knowledgeable AI assistant. "
            f"Answer the following question as accurately as possible: {user_prompt}"
        )
    else:
        return user_prompt

# Chat input
if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    enriched_prompt = enrich_prompt(prompt, prompt_technique)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            response = model.generate_content(
                enriched_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                ),
                stream=True
            )
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Error: Could not generate response."
    st.session_state.messages.append({"role": "assistant", "content": full_response})