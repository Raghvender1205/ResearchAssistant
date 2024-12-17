import streamlit as st
import asyncio
from research_agent import research

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Research Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

def run_research(user_input):
    return asyncio.run(research(user_input))

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(prompt)
    
    with st.spinner('Processing your request...'):
        bot_response = run_research(prompt)

        st.session_state.messages.append({"role": "bot", "content": bot_response})

        # Display bot message in chat message container
        with st.chat_message("bot", avatar="👩‍💻"):
            st.markdown(bot_response)
