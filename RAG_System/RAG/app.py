import streamlit as st

from query_data import process_user_input

def main():
    st.set_page_config(page_title="SSL RAG System", page_icon="⚖️")
    st.header("Chat with the SSL RAG system ⚖️")

    # 2. Initialize Session State to store the conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            # Start with a default Arabic greeting since the documents are in Arabic
            {"role": "assistant", "content": "مرحباً! كيف يمكنني مساعدتك في قانون الضمان الاجتماعي لعام 2014؟"}
        ]

    # 3. Display all previous messages on the screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Handle new user input using the dedicated chat_input widget
    if prompt := st.chat_input("Ask a question about the Social Security Law..."):
        
        # Instantly display the user's question in the UI
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save the user's question to the session memory
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display the response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    history_to_pass = st.session_state.messages[:-1]

                    # Call the new routing function!
                    response_text = process_user_input(prompt, history_to_pass)
                    
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()