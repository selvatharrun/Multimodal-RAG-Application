# 🚀 Multimodal RAG Application with FastAPI & Streamlit

## **Complete Setup, Configuration & Troubleshooting Guide**

---

## 📋 **Table of Contents**
1. [Project Overview](#-project-overview)
2. [Installation & Setup](#-installation--setup)
3. [Configuration Guide](#-configuration-guide)
4. [API Documentation](#-api-documentation)
5. [Features & Capabilities](#-features--capabilities)
6. [Debugging & Troubleshooting](#-debugging--troubleshooting)
7. [Common Errors & Solutions](#-common-errors--solutions)
8. [Environment Variables](#-environment-variables)

---

## 🎯 **Project Overview**

This is a **Multimodal Retrieval-Augmented Generation (RAG)** application that combines:

- **FastAPI Backend**: Two separate APIs for document processing and chat functionality
- **Streamlit Frontend**: Interactive web interface with multiple pages
- **Multiple AI Models**: Support for Azure OpenAI, Google Gemini, Claude, Qwen, and more
- **Advanced OCR**: Multiple OCR methods including Tesseract, Florence-2, Google Vision, Claude Vision
- **Intelligent Search**: BM25, Qdrant embeddings, and Reciprocal Rank Fusion (RRF)

### **Architecture Overview**
```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Streamlit UI      │ ── │   FastAPI Backend    │ ── │   AI Models & OCR   │
│  (Port 8501)        │    │  (Ports 8000/8001)  │    │  (External APIs)    │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

---

## 🛠️ **Installation & Setup**

### **Prerequisites**
- **Python 3.11+** (Tested with 3.13.5)
- **Windows OS** (Current configuration)
- **Git** (for cloning)

### **Step 1: Clone and Setup Environment**
```powershell
# Clone the repository
git clone <repository-url>
cd Multimodal-RAG-Application-with-FastAPI--main

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# If execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

### **Step 2: Install Dependencies**
```powershell
# Install all required packages
pip install -r requirements.txt

# Key packages include:
# - streamlit==1.48.0
# - fastapi==0.116.1
# - transformers==4.53.3
# - langchain==0.3.27
# - torch==2.6.0
# - torchvision==0.21.0
# - pytesseract==0.3.13
# - openai, google-generativeai, anthropic
```

### **Step 3: Install Tesseract OCR**
```powershell
# Option 1: Using Windows Package Manager (Recommended)
winget install --id UB-Mannheim.TesseractOCR

# Option 2: Using Chocolatey
choco install tesseract -y

# Option 3: Manual download from:
# https://github.com/UB-Mannheim/tesseract/releases
```

### **Step 4: Verify Installation**
```powershell
# Test Tesseract
tesseract --version

# Test Python packages
python -c "import streamlit, torch, transformers; print('All dependencies OK!')"
```

---

## ⚙️ **Configuration Guide**

### **Main Configuration File: `florence2/config.properties`**

```properties
[azure]
api_key=YOUR_AZURE_OPENAI_API_KEY
endpoint=YOUR_AZURE_ENDPOINT
deployment=YOUR_DEPLOYMENT_NAME
version=2024-05-01-preview

[aws]
access_key_id=YOUR_AWS_ACCESS_KEY
secret_access_key=YOUR_AWS_SECRET_KEY
region=us-east-1

[google]
api_key=YOUR_GOOGLE_API_KEY

[pytesseract]
file_path=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### **🔑 Where to Get API Keys**

#### **Azure OpenAI**
1. Go to [Azure Portal](https://portal.azure.com)
2. Create/Access Azure OpenAI resource
3. Copy **Key**, **Endpoint**, and **Deployment Name**
4. Update `[azure]` section in config.properties

#### **Google Gemini**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create new API key
3. Update `[google]` section: `api_key=YOUR_KEY_HERE`

#### **AWS (Claude)**
1. Go to [AWS Console](https://console.aws.amazon.com)
2. Access IAM → Create access key
3. Enable Bedrock service
4. Update `[aws]` section with credentials

### **🗂️ File Structure Overview**
```
project-root/
├── florence2/                    # Main application directory
│   ├── main.py                   # FastAPI server (Port 8000) - KO generation
│   ├── chatapi.py                # FastAPI server (Port 8001) - Chat & OCR
│   ├── mainpage.py               # Streamlit main page
│   ├── config.properties         # 🔑 API keys configuration
│   │
│   ├── pages/                    # Streamlit pages
│   │   ├── Docchat.py           # Document chat interface
│   │   ├── Image_analysis.py     # Florence-2 image analysis
│   │   ├── kocreation.py        # Knowledge Object creation
│   │   └── imports/             # Supporting modules
│   │
│   ├── API/                     # Backend API modules
│   │   ├── sama_updated.py      # Document processing (PDF, DOCX, PPTX)
│   │   ├── florence.py          # Florence-2 vision model
│   │   ├── googlevision.py      # Google Vision API
│   │   ├── caludeocr.py         # Claude Vision API  
│   │   ├── open.py              # OpenAI Vision API
│   │   ├── getllm.py            # LLM model selection
│   │   └── searchmethods.py     # RAG search methods
│   │
│   └── TESTS/                   # Test files and notebooks
│
├── venv/                        # Python virtual environment
├── requirements.txt             # Python dependencies
├── run_streamlit.bat           # Quick start script
└── TESSERACT_SETUP.md          # Tesseract installation guide
```

---

## 📡 **API Documentation**

### **Server 1: Main API (Port 8000)**
**Purpose**: Knowledge Object generation from documents

#### **Endpoints:**
- **POST** `/upload-file/` - Generate Knowledge Objects

**Request:**
```python
import requests

url = "http://localhost:8000/upload-file/"
data = {
    "model_name": "google",      # Options: "google", "azureai", "claude3-sonnet", "qwen", "nvidia"
    "ocr_method": "tesseract"    # Options: "tesseract", "openai", "florence", "google", "claude"
}
files = {"file": ("document.pdf", open("document.pdf", "rb"))}
response = requests.post(url, data=data, files=files)
```

**Response:**
```json
{
    "filename": "document.pdf",
    "KO_Article": {
        "Query": "Root cause description",
        "Symptoms": "Observable signs and behaviors",
        "Short_description": "Brief summary",
        "Long_description": "Detailed enhanced description", 
        "Causes": "Underlying reasons",
        "Resolution_note": "Step-by-step solution",
        "Relevancy": "Relevancy percentage [0-100]%"
    }
}
```

### **Server 2: Chat API (Port 8001)**
**Purpose**: Document chat and text extraction

#### **Endpoints:**

##### **1. POST** `/extract_text/` - Extract text from documents
```python
data = {"ocr_method": "tesseract"}
files = {"file": ("document.pdf", file_content)}
response = requests.post("http://localhost:8001/extract_text/", data=data, files=files)
```

##### **2. POST** `/search_and_respond/` - Chat with documents
```python
import json

data = {
    "session_uuid": "123e4567-e89b-12d3-a456-426614174000",
    "extracted_text": "Document content here...",
    "prompt": "What is this document about?",
    "model_name": "google",
    "search_method": "Embedding + Qdrant",
    "conversation_history": []
}
response = requests.post("http://localhost:8001/search_and_respond/", json=data)
```

---

## 🎨 **Features & Capabilities**

### **🖼️ Multimodal Understanding**
- **Text Processing**: PDF, DOCX, PPTX documents
- **Image Analysis**: Florence-2 vision model for detailed image understanding
- **OCR Methods**: 
  - Tesseract (free, local)
  - Google Vision API (cloud, accurate)
  - OpenAI GPT-4 Vision (cloud, intelligent)
  - Claude Vision (cloud, advanced)
  - Florence-2 (local, efficient)

### **🧠 AI Models Supported**
- **Azure OpenAI**: GPT-4, GPT-3.5-turbo
- **Google Gemini**: gemini-1.5-flash, gemini-pro
- **Claude**: claude-3-sonnet via AWS Bedrock
- **Qwen**: Local model via Ollama
- **Nvidia**: NeMo models via Ollama

### **🔍 Advanced Search Methods**
- **BM25**: Traditional keyword-based search
- **Qdrant Embeddings**: Semantic vector search
- **Reciprocal Rank Fusion (RRF)**: Combines BM25 + embeddings

### **📱 Streamlit Interface**
1. **Main Page**: Welcome and feature overview
2. **Document Chat**: Upload docs and chat with AI
3. **Image Analysis**: Florence-2 powered image understanding
4. **Knowledge Object Creation**: Generate structured knowledge articles

---

## 🚀 **Running the Application**

### **Method 1: Quick Start (Recommended)**
```powershell
# Double-click the batch file:
.\run_streamlit.bat
```

### **Method 2: Manual Startup**
```powershell
# Terminal 1: Start main API server
cd florence2
python main.py

# Terminal 2: Start chat API server  
cd florence2
python chatapi.py

# Terminal 3: Start Streamlit UI
streamlit run florence2/mainpage.py
```

### **Method 3: Using VS Code Tasks**
1. Open in VS Code
2. `Ctrl+Shift+P` → "Tasks: Run Task"
3. Select:
   - "Start FastAPI Server (Port 8000)"
   - "Start Chat API Server (Port 8001)" 
   - "Run Streamlit App"

### **🌐 Access Points**
- **Streamlit UI**: http://localhost:8501
- **Main API Docs**: http://localhost:8000/docs
- **Chat API Docs**: http://localhost:8001/docs

---

## 🐛 **Debugging & Troubleshooting**

### **🔍 Quick Diagnostics**
```powershell
# Check if servers are running
netstat -an | findstr "8000 8001 8501"

# Test API connectivity
curl http://localhost:8000/docs
curl http://localhost:8001/docs

# Check Python environment
python -c "import sys; print(sys.executable)"
python -c "import streamlit, fastapi, torch; print('All OK')"
```

### **📊 Server Status Checker**
Create `debug.py`:
```python
import requests
import socket

def check_server(port, name):
    # Check if port is open
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(('localhost', port))
        port_open = result == 0
    
    # Check HTTP response
    try:
        response = requests.get(f"http://localhost:{port}/docs", timeout=5)
        http_ok = response.status_code == 200
    except:
        http_ok = False
    
    print(f"{name} (Port {port}): Port {'✅' if port_open else '❌'} | HTTP {'✅' if http_ok else '❌'}")

check_server(8000, "Main API")
check_server(8001, "Chat API") 
check_server(8501, "Streamlit")
```

---

## ❗ **Common Errors & Solutions**

### **1. ConnectionRefusedError: [WinError 10061]**
**Problem**: FastAPI servers not running

**Solutions**:
```powershell
# Check what's using the ports
netstat -ano | findstr "8000"
netstat -ano | findstr "8001"

# Kill existing processes if needed
taskkill /F /PID <process_id>

# Restart servers
cd florence2
python main.py &
python chatapi.py &
```

### **2. TesseractNotFoundError**
**Problem**: Tesseract not installed or wrong path

**Solutions**:
```powershell
# Install Tesseract
winget install --id UB-Mannheim.TesseractOCR

# Verify installation
tesseract --version

# Update config.properties with correct path
# [pytesseract]
# file_path=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### **3. ImportError: No module named 'xxx'**
**Problem**: Missing Python dependencies

**Solutions**:
```powershell
# Reinstall requirements
pip install -r requirements.txt

# Install specific missing packages
pip install streamlit fastapi transformers torch

# Check virtual environment is activated
echo $env:VIRTUAL_ENV  # Should show venv path
```

### **4. API Key Authentication Errors**
**Problem**: Invalid or missing API keys

**Solutions**:
1. **Azure OpenAI**: 
   - Check endpoint URL format: `https://YOUR-RESOURCE.openai.azure.com/`
   - Verify deployment name matches your Azure resource
   - Ensure API version is correct: `2024-05-01-preview`

2. **Google API**:
   - Regenerate key at [AI Studio](https://makersuite.google.com/app/apikey)
   - Enable required APIs in Google Cloud Console

3. **AWS/Claude**:
   - Check IAM permissions for Bedrock access
   - Verify region supports Claude models

### **5. Module Import Errors in Florence/Vision**
**Problem**: Missing computer vision dependencies

**Solutions**:
```powershell
# Install missing packages
pip install einops timm pillow opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA support (if you have GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **6. Streamlit Pages Not Loading**
**Problem**: Page routing or import issues

**Solutions**:
```powershell
# Check file structure
ls florence2/pages/

# Verify imports in pages
python -c "from florence2.pages.imports import florencevlm"

# Run specific page directly
streamlit run florence2/pages/Docchat.py
```

---

## 🔧 **Environment Variables & Configuration**

### **Windows Environment Setup**
```powershell
# Set environment variables (optional, config.properties preferred)
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "your-api-key"
$env:GOOGLE_API_KEY = "your-google-key"
```

### **Development vs Production Settings**

#### **Development (Current Setup)**
```python
# FastAPI servers
uvicorn.run(app, host="0.0.0.0", port=8000)  # Accessible from any IP
uvicorn.run(app, host="0.0.0.0", port=8001)

# Streamlit
streamlit run mainpage.py  # Auto-opens browser
```

#### **Production Considerations**
```python
# Use specific host for security
uvicorn.run(app, host="127.0.0.1", port=8000)

# Use environment variables for sensitive data
import os
api_key = os.getenv('AZURE_OPENAI_API_KEY')

# Enable HTTPS and authentication
```

---

## 🎯 **Performance Optimization**

### **Model Loading**
- **Florence-2**: Cached with `@st.cache_resource`
- **Embeddings**: HuggingFace models cached locally
- **LLMs**: Connection pooling for API calls

### **Memory Management**
```python
# Clear GPU memory after processing
import torch
torch.cuda.empty_cache()

# Use smaller models for testing
# model_name = "paraphrase-MiniLM-L6-v2"  # Smaller embedding model
```

### **Scaling Recommendations**
- **Database**: Replace in-memory storage with persistent DB
- **Queue System**: Add Redis/Celery for background processing
- **Load Balancer**: Use nginx for multiple server instances
- **Monitoring**: Add logging and health checks

---

## 📚 **Additional Resources**

### **Documentation Links**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

### **Model Documentation**
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Google Gemini](https://ai.google.dev/docs)
- [Qwen Models](https://huggingface.co/Qwen)

### **Troubleshooting Resources**
- Check logs in terminal where servers are running
- Use FastAPI `/docs` endpoints for API testing
- Streamlit debug mode: `streamlit run --server.enableWebsocketCompression false`

---

## 🆘 **Getting Help**

### **When Things Break:**

1. **Check Server Status**: Are all 3 servers running? (8000, 8001, 8501)
2. **Verify API Keys**: Test each service independently
3. **Check Dependencies**: `pip list` and compare with requirements.txt
4. **Review Logs**: Terminal output often shows the exact error
5. **Test Minimal**: Start with basic functionality first

### **Debug Priority Order:**
1. ✅ Virtual environment activated
2. ✅ Dependencies installed  
3. ✅ Tesseract installed and configured
4. ✅ API keys set in config.properties
5. ✅ FastAPI servers running (8000, 8001)
6. ✅ Streamlit server running (8501)

---

## 🎉 **Success Checklist**

When everything is working, you should see:

- ✅ **Main API**: http://localhost:8000/docs shows Swagger UI
- ✅ **Chat API**: http://localhost:8001/docs shows endpoints
- ✅ **Streamlit**: http://localhost:8501 shows welcome page
- ✅ **Upload**: Can upload PDF/DOCX files
- ✅ **OCR**: Text extraction works with Tesseract
- ✅ **Chat**: Document Q&A responds appropriately
- ✅ **KO Generation**: Knowledge objects are created
- ✅ **Image Analysis**: Florence-2 processes images

**🚀 You're ready to explore the multimodal AI capabilities!**

---

*Last updated: August 2025 | For issues, check the troubleshooting section above*
