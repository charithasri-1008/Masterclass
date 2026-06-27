import streamlit as st

st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖")

st.title("🤖 HR Policy Chatbot")
st.write("Ask questions about the HR policy documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask your HR question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_bot(question)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )