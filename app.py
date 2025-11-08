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

if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "enhanced_prompt" not in st.session_state:
    st.session_state.enhanced_prompt = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to enrich prompt based on selected technique
def enrich_prompt(user_prompt, technique, model, temperature, top_p, top_k):
    if technique == "None":
        return user_prompt

    enhancement_instruction = ""
    if technique == "Few-shot":
        enhancement_instruction = (
            "Generate a few-shot prompt based on the user's input. "
            "Provide 2-3 relevant examples before the user's actual prompt to guide the model. "
            "The examples should be in a Question-Answer format if applicable, or demonstrate the desired output style."
        )
    elif technique == "Chain-of-Thought":
        enhancement_instruction = (
            "Generate a Chain-of-Thought prompt based on the user's input. "
            "Add a prefix like 'Let's think step by step.' and instruct the model to explain its reasoning before providing the final answer."
        )
    elif technique == "Persona-based":
        enhancement_instruction = (
            "Generate a persona-based prompt based on the user's input. "
            "Assign a specific persona (e.g., 'You are a helpful and knowledgeable AI assistant', 'You are a seasoned data scientist') "
            "to the model that is relevant to the user's prompt, and then present the user's prompt."
        )

    full_enhancement_prompt = (
        f"The user wants to apply the '{technique}' prompt engineering technique to their prompt. "
        f"Here is the user's original prompt: '{user_prompt}'.\n\n"
        f"Your task is to generate an enhanced prompt by applying the '{technique}' technique. "
        f"{enhancement_instruction}\n\n"
        "Provide only the enhanced prompt, without any additional conversational text."
    )

    st.write(f"Debug: Selected technique in enrich_prompt: {technique}, temperature {temperature}, top_p {top_p}, top_k {top_k}")
    try:
        response = model.generate_content(
            full_enhancement_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        )
        return response.text
    except Exception as e:
        st.error(f"Error generating enhanced prompt with Gemini: {e}")
        return user_prompt # Fallback to original prompt on error

# User input area
user_input_container = st.container()
with user_input_container:
    st.session_state.user_input = st.text_area("Enter your prompt here...", value=st.session_state.user_input, height=150, key="user_prompt_input")
    
    col1, col2 = st.columns([3, 8]) # Adjusted column ratio to give more width to the first button
    with col1:
        if st.button("Enhance Prompt"):
            st.session_state.enhanced_prompt = enrich_prompt(st.session_state.user_input, prompt_technique, model, temperature, top_p, top_k)
            st.rerun()
    with col2:
        if st.button("Send Prompt"):
            prompt_to_send = st.session_state.user_input
            if prompt_to_send:
                st.session_state.messages.append({"role": "user", "content": prompt_to_send})
                with st.chat_message("user"):
                    st.markdown(prompt_to_send)

                enriched_prompt = enrich_prompt(prompt_to_send, prompt_technique, model, temperature, top_p, top_k)

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
                st.session_state.user_input = "" # Clear input after sending
                st.rerun()

# Display enhanced prompt for approval
if st.session_state.enhanced_prompt:
    st.subheader("Suggested Enhanced Prompt")
    st.text_area("Review and Approve", value=st.session_state.enhanced_prompt, height=150, key="enhanced_prompt_review")
    
    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("Approve Enhanced Prompt"):
            st.session_state.user_input = st.session_state.enhanced_prompt
            st.session_state.enhanced_prompt = ""
            st.rerun()
    with col_reject:
        if st.button("Reject Enhanced Prompt"):
            st.session_state.enhanced_prompt = ""
            st.rerun()