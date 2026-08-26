import streamlit as st
import ollama

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="My AI Chatbot",
)

# -----------------------------
# Title
# -----------------------------

st.title("🤖 My AI Chatbot")

# -----------------------------
# System prompt
# -----------------------------

SYSTEM_PROMPT = """
You are a helpful AI programming tutor.
Explain programming concepts clearly and simply.
Use examples when useful.
If the user asks a programming question, provide a clear explanation.
"""

# -----------------------------
# Initialize conversation memory
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("Model: `llama3.2`")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        st.rerun()

# -----------------------------
# Display previous conversation
# -----------------------------

for message in st.session_state.messages:

    # Don't display system prompt
    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat input
# -----------------------------

user_input = st.chat_input("Ask me anything...")

# -----------------------------
# Process user message
# -----------------------------

if user_input:

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate AI response
    with st.chat_message("assistant"):

        response = ollama.chat(
            model="llama3.2",
            messages=st.session_state.messages,
            stream=True
        )

        # Stream the response
        bot_response = st.write_stream(
            chunk["message"]["content"]
            for chunk in response
        )

    # Store AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_response
        }
    )