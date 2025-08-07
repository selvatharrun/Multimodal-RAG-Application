from fastapi import FastAPI, File, UploadFile, Form,Body
from fastapi.responses import JSONResponse
from API.sama_updated import read_docx, process_pdf, convert_to_markdown, extract_text_from_pptx
from API.searchmethods import qdrant_search, ReciprocalRankFusion, bm25s_search
from API.open import process_file_with_gpt_vision
from API.florence import process_file_with_florence
from API.googlevision import process_file_with_google_vision
from API.caludeocr import process_file_with_claude
from API.getllm import get_llm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
import bm25s
import Stemmer
import os
from uuid import UUID
from typing import List, Dict

app = FastAPI()

# In-memory storage for conversation histories by UUID
conversation_histories: Dict[UUID, List[Dict[str, str]]] = {}

# Helper function to initialize conversation history for a new session UUID
def get_or_create_conversation(session_uuid: UUID):
    if session_uuid not in conversation_histories:
        conversation_histories[session_uuid] = []
    return conversation_histories[session_uuid]

# Route 1: Text Extraction with OCR
@app.post("/extract_text/")
async def extract_text_api(
    file: UploadFile = File(...),
    ocr_method: str = Form("tesseract")
):
    # Save the uploaded file
    uploads_folder = "uploads"
    os.makedirs(uploads_folder, exist_ok=True)
    file_path = os.path.join(uploads_folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    if ocr_method == "tesseract":
        # Process the file based on type
        if file.filename.endswith(".pdf"):
            content = process_pdf(file_path)
        elif file.filename.endswith(".docx"):
            content = read_docx(file_path)
        elif file.filename.endswith(".pptx"):
            content = extract_text_from_pptx(file_path)
        else:
            return JSONResponse(status_code=400, content={"message": "Unsupported file format."})
        all_text = ""
        for item in content:
            all_text += convert_to_markdown(item)

    # Perform OCR
    if ocr_method == "openai":
        all_text = process_file_with_gpt_vision(file_path, uploads_folder, verbose=True)

    elif ocr_method == "florence":
        all_text = process_file_with_florence(file_path, uploads_folder, verbose=True)
        all_text = all_text[0].get("<OCR>")

    elif ocr_method == "google":
        all_text = process_file_with_google_vision(file_path, uploads_folder, verbose=True)

    elif ocr_method == "claude":
        all_text = process_file_with_claude(file_path, uploads_folder, verbose=True)
        all_text = all_text[0].get("text")

    return JSONResponse(all_text)

# Helper functions for search and LLM operations
def load_from_string(text: str):
    document = Document(page_content=text, metadata={"source": "string_input"})
    return [document]

def process_documents_with_qdrant(docs, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    qdrant = Qdrant.from_documents(split_docs, embedding_model, location=":memory:", collection_name="my_documents")
    return qdrant

def init_bm25s_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ".", ""])
    split_docs = text_splitter.split_documents(docs)
    corpus = [{'id': i, 'metadata': doc.metadata, 'text': doc.page_content} for i, doc in enumerate(split_docs)]
    stemmer = Stemmer.Stemmer("english")
    texts = [doc['text'] for doc in corpus]
    corpus_tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever, corpus, stemmer


from fastapi import Body

# Route 2: Search and Generate Response
@app.post("/search_and_respond/")
async def search_and_respond(
    session_uuid: UUID = Body(...),  # Expecting JSON, not Form
    extracted_text: str = Body(...),
    prompt: str = Body(...),
    model_name: str = Body("google"),
    search_method: str = Body("Embedding + Qdrant"),
    conversation_history: list[dict] = Body(default=[])
):
     # Use the provided conversation history or initialize a new one if not available.
    if not conversation_history:
        conversation_history = get_or_create_conversation(session_uuid)
    else:
        conversation_histories[session_uuid] = conversation_history

    structured_prompt = f"""
    You are an AI assistant helping a user with classifying questions between general or document-based inquiries.

    The user has asked: '{prompt}'.

    First, determine if this question is general, like a greeting or non-document-related query. 
    For general questions (e.g., 'hi', 'hello', or questions not tied to specific documents or refering to previous conversation), respond without needing document context.

    For document-specific questions, guide the user to feed the relevant document, to extract context from.

    strictly respond with a single word, if u think the user does not need external document context, to answer that question reply 'no', else reply 'document'.
    """
    llm = get_llm(model_name)
    
    # Generate an initial response to assess if document context is necessary
    preliminary_response = llm.invoke(structured_prompt)
    res1 = preliminary_response.content if hasattr(preliminary_response, 'content') else str(preliminary_response)

    if "document" in res1.lower():  
        # Load document text and initialize the search method
        doc_texts = load_from_string(extracted_text)
        qdrant = process_documents_with_qdrant(doc_texts) if search_method in ["Embedding + Qdrant", "RRF"] else None
        bm25_retriever, bm25_corpus, bm25_stemmer = init_bm25s_retriever(doc_texts) if search_method in ["BM25S", "RRF"] else (None, None, None)
        
        # Retrieve relevant document context based on the search method
        doc_context = ""
        if search_method == "Embedding + Qdrant":
            doc_context = qdrant_search(prompt, qdrant)
        elif search_method == "BM25S":
            doc_context = bm25s_search(prompt, bm25_retriever, bm25_stemmer, bm25_corpus)
        elif search_method == "RRF":
            bm25_results = bm25s_search(prompt, bm25_retriever, bm25_stemmer, bm25_corpus)
            embedding_results = qdrant_search(prompt, qdrant)
            rrf = ReciprocalRankFusion()
            doc_context = rrf.fuse([bm25_results, embedding_results], top_n=3)
        else:
            return JSONResponse(status_code=400, content={"message": "Unsupported search method."})

        # Build the final prompt using document context and conversation history
        conversation_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in conversation_history
        )
        final_prompt = f"""
        Conversation history: {conversation_text}

        Document context: {doc_context}

        User question: '{prompt}'

        Answer based on the document context and the ongoing conversation.
        """
    else:
        # For general questions, generate a response without document context
        final_prompt = f"""
        User question: '{prompt}'

        Since this is a general question, provide an answer without referencing specific document content.
        """
    
    # Invoke the language model
    response = llm.invoke(final_prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Append the user's question and assistant's response to the conversation history
    conversation_history.append({"role": "user", "content": prompt})
    conversation_history.append({"role": "assistant", "content": response_text})

    # Return the response as JSON
    return JSONResponse(content={"response": response_text})

if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8001)
