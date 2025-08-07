import os
import io
import tempfile
import streamlit as st
from transformers import pipeline
import base64
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for PDF processing
import pdfplumber  # For table extraction from PDFs
from streamlit_chat import message
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import base64
import tempfile
import bm25s    
import Stemmer
import heapq
from pages.imports.sama_updated import read_docx, process_pdf, convert_to_markdown, extract_text_from_pptx
from pages.imports.searchmethods import qdrant_search, ReciprocalRankFusion, bm25s_search, KO
import ollama  # Import the ollama client

# Set up pytesseract for OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\sselva\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

@st.cache_resource
def init_ollama_model():
    llm = Ollama(model="qwen2.5:1.5b")
    return llm

# Function to display PDF
def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Custom function to load a string as a Document object
def load_from_string(text: str):
    document = Document(page_content=text, metadata={"source": "string_input"})
    return [document]

@st.cache_resource
def process_documents_with_qdrant(_docs, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    qdrant = Qdrant.from_documents(split_docs, embedding_model, location=":memory:", collection_name="my_documents")
    return qdrant

# BM25S initialization
@st.cache_resource
def init_bm25s_retriever(_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ".", "",]
    )
    split_docs = text_splitter.split_documents(docs)
    corpus = [{'id': i, 'metadata': doc.metadata, 'text': doc.page_content} for i, doc in enumerate(split_docs)]
    stemmer = Stemmer.Stemmer("english")
    texts = [doc['text'] for doc in corpus]
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever, corpus, stemmer

# Knowledge Object generation function
def generate_knowledge_object(doc_context: str, prompt: str) -> str:
    return f"""
    You are an AI assistant tasked with generating a Knowledge Object based on the given context and user input.
    Context: '{doc_context}'
    Use this context to generate a detailed KO in the following format:
    
    - Short Description: (Explain the root cause of the problem)
    - Symptoms: (List observable signs or behaviors indicating the issue)
    - Long Description: (Provide a detailed description of the problem or issue in 50 words)
    - Causes: (Identify the factors that led to this issue)
    - Resolution Note: (Give a step-by-step resolution for the problem, covering all scenarios)
    
    Question: '{prompt}'
    """

def run(model: str, doc_context: str, question: str):
    client = ollama.Client()

    # Initialize conversation with a user query
    messages = [{"role": "user", "content": question}]

    # First API call: Send the query and function description to the model
    response = client.chat(
        model=model,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "generate_knowledge_object",
                    "description": "Generate a Knowledge Object based on context and user input",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_context": {
                                "type": "string",
                                "description": "Document context relevant to the KO generation"
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The user's question or input for generating the KO"
                            }
                        },
                        "required": ["doc_context", "prompt"],
                    },
                },
            }
        ],
    )

    # Add the model's response to the conversation history
    messages.append(response["message"])

    if not response["message"].get("tool_calls"):
        return response["message"]["content"]
    
    # Process function calls made by the model
    if response["message"].get("tool_calls"):
        available_functions = {
            "generate_knowledge_object": generate_knowledge_object,
        }

        for tool in response["message"]["tool_calls"]:
            function_to_call = available_functions[tool["function"]["name"]]
            function_args = tool["function"]["arguments"]
            function_response = function_to_call(**function_args)

            # Add function response to the conversation
            messages.append(
                {
                    "role": "tool",
                    "content": function_response,
                }
            )

    # Second API call: Get final response from the model
    final_response = client.chat(model=model, messages=messages)

    return final_response["message"]["content"]

# Initialize app state if not already done
if 'app' not in st.session_state:
    st.session_state['app'] = []

# Main Streamlit app logic
st.title("Chat with Document (PDF/DOCX) using Qwen 2.5")
st.caption("Choose your search method: Embedding+Qdrant or BM25S")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'llm' not in st.session_state:
    st.session_state.llm = init_ollama_model()

# File upload and processing
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file:
        display_pdf(uploaded_file)
        search_method = st.selectbox("Choose Search Method", ["Embedding + Qdrant", "Corpus + BM25S", "Hybrid Search (RRF)"])
        if st.button("Process and Convert to Markdown"):
            with st.spinner("Processing the document..."):
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                if file_path.endswith(".pdf"):
                    content = process_pdf(file_path)
                elif file_path.endswith(".docx"):
                    content = read_docx(file_path)
                elif file_path.endswith(".pptx"):
                    content = extract_text_from_pptx(file_path)
                else:
                    st.warning("Unsupported file type.")
                    content = []

                markdown_text = ""
                for item in content:
                    markdown_text += convert_to_markdown(item)

                st.session_state.markdown_text = markdown_text
                docs = load_from_string(st.session_state.markdown_text)

                # Process documents based on chosen search method
                if search_method == "Embedding + Qdrant":
                    st.session_state.qdrant = process_documents_with_qdrant(docs)
                elif search_method == "Corpus + BM25S":
                    st.session_state.bm25s_retriever, st.session_state.bm25s_corpus, st.session_state.bm25s_stemmer = init_bm25s_retriever(docs)
                elif search_method == "Hybrid Search (RRF)":
                    docs = load_from_string(st.session_state.markdown_text)
                    st.session_state.qdrant = process_documents_with_qdrant(docs)
                    st.session_state.bm25s_retriever, st.session_state.bm25s_corpus, st.session_state.bm25s_stemmer = init_bm25s_retriever(docs)

# Chat interface: Display previous messages
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(i))

if prompt := st.chat_input("Ask a question about the document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    with st.spinner("Thinking..."):
        # Assuming doc_context should only be fetched if the LLM decides to use it
        doc_context = ""  # Initialize an empty document context
        
        # Create a structured prompt that asks the LLM to determine if the document context is needed
        structured_prompt = f"""
        You are an intelligent AI assistant. The user has asked: '{prompt}'.
        
        Assess if this is a general question or if it requires processing the document context. 
        If it's a general question like 'hi', 'hello', or other non-document queries, answer it without using the document context.

        If the query seems to be document-related, you can request the relevant document context and include it in your response.
        """

        # Generate a temporary response from the LLM to assess if document context is needed
        preliminary_response = st.session_state.llm(structured_prompt)

        # Check if the LLM determined that the document context is necessary
        if "document" in preliminary_response.lower():  # Simple check; you can refine this
            if search_method == "Embedding + Qdrant":
                doc_context = qdrant_search(prompt, st.session_state.qdrant)
            elif search_method == "Corpus + BM25S":
                doc_context = bm25s_search(prompt, st.session_state.bm25s_retriever, st.session_state.bm25s_stemmer, st.session_state.bm25s_corpus)
            elif search_method == "Hybrid Search (RRF)":
                bm25_results = bm25s_search(prompt, st.session_state.bm25s_retriever, st.session_state.bm25s_stemmer, st.session_state.bm25s_corpus)
                embedding_results = qdrant_search(prompt, st.session_state.qdrant)
                
                # Fuse results using Reciprocal Rank Fusion
                rrf = ReciprocalRankFusion()
                doc_context = rrf.fuse([bm25_results, embedding_results], top_n=3)

        # If the document context is fetched, append it to the final structured prompt
        if doc_context:
            print("need context############")
            response = run("qwen2.5:1.5b", doc_context, prompt)
        else:
            # If no document context is required, just answer the question normally
            final_prompt = f"""
            The user asked: '{prompt}'.
            
            Since this is a general question, answer it without referencing any specific document context.
            """
            response = st.session_state.llm(final_prompt)

        # Append the response to the chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response)
