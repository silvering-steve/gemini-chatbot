import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold

import streamlit as st

import asyncio


def main():
    vertexai.init(
        project="learn-408903",  # st.secrets["projects_id"],
        location="asia-southeast1"  # st.secrets["locations"]
    )

    st.header("Gemini Chat", divider="gray")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat" not in st.session_state:
        model = GenerativeModel("gemini-pro")
        st.session_state.chat = model.start_chat()

    if "config" not in st.session_state:
        config = None

    if "safe" not in st.session_state:
        safe = None

    with st.sidebar:
        st.title("Settings")

        settings_tab, prompt_tab = st.tabs(["Settings", "Prompt"])

        with settings_tab:
            token = st.text_input("Max Token", value=2048)

            col1_1, col1_2, col1_3 = st.columns(3)

            with col1_1:
                temp = st.slider("Temperature", 0.0, 1., value=0.5)

            with col1_2:
                top_k = st.slider("Top K", min_value=0, max_value=40, value=3)

            with col1_3:
                top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.7)

        with prompt_tab:
            context = st.text_area("Context", value="You are an helpful assistant")
            example = st.text_area("Example", value="""USER: Hello! \nAI: Howdy!
            """)

            if st.button("Chat", type="primary"):
                st.session_state.config = {
                    "max_output_tokens": int(token),
                    "temperature": float(temp),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                }
                st.session_state.safety = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                }

                st.session_state.chat.send_message(
                    f"Context: {context} Examples: {example}",
                    generation_config=st.session_state.config,
                    safety_settings=st.session_state.safety
                )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            responds = st.session_state.chat.send_message(
                prompt,
                generation_config=st.session_state.config,
                safety_settings=st.session_state.safety
            ).text

            st.markdown(responds)
            st.session_state.messages.append({"role": "assistant", "content": responds})


if __name__ == '__main__':
    st.set_page_config(
        page_title="Chatbot",
    )
    main()
