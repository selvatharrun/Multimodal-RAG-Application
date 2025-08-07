import os
import streamlit as st
import requests
from streamlit_chat import message
import uuid
import time

# Define API endpoints
OCR_API_URL = "http://localhost:8001/extract_text/"
LLM_API_URL = "http://localhost:8001/search_and_respond/"

# Styling for Streamlit app
st.markdown('''
    <style>
        [data-testid="stBottomBlockContainer"] {
            width: 100%;
            padding: 1rem 2rem 1rem 2rem;
            min-width: auto;
            max-width: initial;
        }
        [data-testid="column"]{
            position:sticky;
            align-content:center;
            padding:1rem;
        }
        [data-testid="stAppViewBlockContainer"]{
            width: 100%;
            padding: 2rem;
            min-width: auto;
            max-width: initial;
        }
        [data-testid="stVerticalBlock"]{
            gap:0.6rem;
        }
        .uploadedFiles {
            display: none;
        }   
    </style>
''', unsafe_allow_html=True)

# Main column layout
col1, col2 = st.columns([9, 1])
with col1:
    st.title("Ask DOC")
    st.subheader("chat with (pdf/docx)")
with col2:
    with st.popover("⚙️"):
        st.header("Document and Settings")
        model_name = st.selectbox("Select LLM", ["google", "azureai", "claude3-sonnet", "qwen", "nvidia"])
        ocr_method = st.selectbox("Select OCR Method", ["tesseract", "openai", "florence", "google", "claude"])
        search_method = st.selectbox("Select Search Method", ["Embedding + Qdrant", "BM25S", "RRF"])

# Initialize session variables
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}  # Stores all conversations by UUID
if 'current_convo_id' not in st.session_state:
    st.session_state.current_convo_id = None  # Active conversation ID
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}  # Store file names and their OCR status

# Sidebar for managing conversations
with st.sidebar:
    st.header("Manage Conversations")
    
    # List all existing conversations in the session
    for convo_id in st.session_state.conversations:
        convo_title = f"Conversation {convo_id[-4:]}"
        if st.button(convo_title):
            st.session_state.current_convo_id = convo_id
    
    # Button to start a new conversation
    if st.button("Start New Conversation"):
        new_convo_id = str(uuid.uuid4())
        st.session_state.conversations[new_convo_id] = []
        st.session_state.current_convo_id = new_convo_id

    # File upload and file selection
    uploaded_files = st.file_uploader("Upload files (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files[file.name] = {
                    "content": file,
                    "ocr_text": "",
                    "selected": False
                }

    # Expander for file selection
    with st.expander("Uploaded Files"):
        for file_name, file_data in st.session_state.uploaded_files.items():
            st.session_state.uploaded_files[file_name]["selected"] = st.checkbox(file_name, key=f"{file_name}_checkbox")

            # OCR processing if selected and not already done
            if st.session_state.uploaded_files[file_name]["selected"] and not file_data["ocr_text"]:
                files = {"file": (file_data["content"].name, file_data["content"].getvalue())}
                ocr_data = {"ocr_method": ocr_method}
                ocr_response = requests.post(OCR_API_URL, files=files, data=ocr_data)
                
                if ocr_response.status_code == 200:
                    st.session_state.uploaded_files[file_name]["ocr_text"] = ocr_response.json()
                else:
                    st.error(f"OCR error for {file_name}: {ocr_response.text}")

# Check if a conversation is selected
if st.session_state.current_convo_id is None:
    st.warning("Start a new conversation or select an existing one.")
else:
    # Gather OCR text from selected files
    combined_ocr_text = "\n".join(
        file_data["ocr_text"]
        for file_data in st.session_state.uploaded_files.values()
        if file_data["selected"]
    )

    # Display chat messages for the current conversation
    current_conversation = st.session_state.conversations[st.session_state.current_convo_id]
    for i, msg in enumerate(current_conversation):
        message(msg["content"], is_user=msg["role"] == "user", key=f"{st.session_state.current_convo_id}_{i}")

    # Chat input for questions
    if prompt := st.chat_input("Ask a question about the document"):
        current_conversation.append({"role": "user", "content": prompt})
        message(prompt, is_user=True)
        
        # Ensure a document is uploaded and OCR text is available
        if combined_ocr_text:
            llm_data = {
                "session_uuid": st.session_state.current_convo_id,
                "extracted_text": combined_ocr_text,
                "prompt": prompt,
                "model_name": model_name,
                "search_method": search_method,
                "conversation_history": current_conversation
            }

            # API call for LLM response
            with st.spinner("Processing your question..."):
                response = requests.post(LLM_API_URL, json=llm_data)
                if response.status_code == 200:
                    response_data = response.json()
                    assistant_message = response_data.get("response")
                    
                    # Display the assistant's response
                    current_conversation.append({"role": "assistant", "content": assistant_message})
                    message(assistant_message, is_user=False)
                else:
                    st.error("An error occurred: " + response.text)
        else:
            st.warning("Please select a document for processing.")
