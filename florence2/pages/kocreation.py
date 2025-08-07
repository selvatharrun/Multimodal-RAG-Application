import requests
import streamlit as st

# Define the API endpoint URL
url = "http://localhost:8000/upload-file/"  # Replace with your actual FastAPI endpoint

# Streamlit UI
st.set_page_config(page_title="Knowledge Object Generator", layout="wide")

st.title("Knowledge Object Generator")
st.write("Upload multiple files, select an LLM model, specify an OCR method, and generate Knowledge Objects for each file.")

# Sidebar for model and OCR selection
st.sidebar.header("Configuration")
model_name = st.selectbox("Select LLM", ["google","azureai", "claude3-sonnet", "qwen", "nvidia"],index=0)
ocr_method = st.selectbox("Select OCR Method", ["tesseract", "openai", "florence", "google", "claude"],index=0)

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "pptx"], accept_multiple_files=True)

# Placeholder for displaying results
results = []

# Submit button to process files
if st.button("Generate KO Articles"):
    if uploaded_files:
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                # Prepare form data for each file
                data = {
                    "model_name": model_name,
                    "ocr_method": ocr_method,
                }
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                
                # Send request to API
                response = requests.post(url, data=data, files=files)
                
                # Store the response for each file
                if response.status_code == 200:
                    results.append({
                        "filename": uploaded_file.name,
                        "response": response.json()
                    })
                else:
                    results.append({
                        "filename": uploaded_file.name,
                        "error": response.text
                    })

        st.success("All files processed successfully!")
    else:
        st.warning("Please upload at least one file.")

# Display results in expandable sections
if results:
    st.write("### Generated Knowledge Objects")
    for result in results:
        filename = result["filename"]
        if "response" in result:
            with st.expander(f"KO Article for {filename}", expanded=False):
                st.json(result["response"])  # Display the KO article JSON response
        else:
            st.error(f"Error processing {filename}: {result['error']}")

