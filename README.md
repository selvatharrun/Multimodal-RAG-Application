# 🚀 Multimodal RAG Application with FastAPI & Streamlit

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.116+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.48+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A comprehensive **Multimodal Retrieval-Augmented Generation (RAG)** application that combines FastAPI backend with Streamlit frontend, supporting multiple AI models, advanced OCR capabilities, and intelligent document processing.

## ✨ Features

### 🤖 **Multi-Model AI Support**
- **Azure OpenAI** (GPT-4, GPT-3.5-turbo)
- **Google Gemini** (gemini-1.5-flash, gemini-pro)
- **Claude** (claude-3-sonnet via AWS Bedrock)
- **Qwen & Nvidia** (Local models via Ollama)

### 🖼️ **Advanced OCR & Vision**
- **Tesseract OCR** (Free, local processing)
- **Florence-2** (Microsoft's vision model)
- **Google Vision API** (Cloud-based accuracy)
- **OpenAI GPT-4 Vision** (Intelligent understanding)
- **Claude Vision** (Advanced document analysis)

### 🔍 **Intelligent Search**
- **BM25** (Traditional keyword search)
- **Qdrant Embeddings** (Semantic vector search)
- **Reciprocal Rank Fusion** (Hybrid approach)

### 📱 **User Interface**
- **Streamlit Web App** with multiple pages
- **Document Chat** interface
- **Image Analysis** powered by Florence-2
- **Knowledge Object** generation
- **Real-time OCR** processing

## 🏗️ Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Streamlit UI      │ ── │   FastAPI Backend    │ ── │   AI Models & OCR   │
│  (Port 8501)        │    │  (Ports 8000/8001)  │    │  (External APIs)    │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

- **Port 8000**: Main API for Knowledge Object generation
- **Port 8001**: Chat API for document interaction and OCR
- **Port 8501**: Streamlit web interface

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Windows OS (current configuration)
- Git

### 1. Clone and Setup
```bash
git clone https://github.com/selvatharrun/Multimodal-RAG-Application.git
cd Multimodal-RAG-Application

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Tesseract OCR
```powershell
# Using Windows Package Manager (Recommended)
winget install --id UB-Mannheim.TesseractOCR

# Or download from: https://github.com/UB-Mannheim/tesseract/releases
```

### 3. Configure API Keys
```bash
# Copy template and add your API keys
cp florence2/config.properties.template florence2/config.properties
# Edit config.properties with your actual API keys
```

### 4. Run the Application
```bash
# Quick start (all servers)
.\run_streamlit.bat

# Or manually:
# Terminal 1: python florence2/main.py
# Terminal 2: python florence2/chatapi.py  
# Terminal 3: streamlit run florence2/mainpage.py
```

### 5. Access the Application
- **Web UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs & http://localhost:8001/docs

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)** | 📚 Complete setup, API docs, troubleshooting |
| **[TESSERACT_SETUP.md](TESSERACT_SETUP.md)** | 🔧 Tesseract OCR installation guide |
| **config.properties.template** | ⚙️ Configuration template for API keys |

## 🔑 API Key Setup

### Azure OpenAI
1. Visit [Azure Portal](https://portal.azure.com)
2. Create/access Azure OpenAI resource
3. Copy Key, Endpoint, and Deployment Name

### Google Gemini
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create new API key

### AWS Claude
1. Access [AWS Console](https://console.aws.amazon.com)
2. Create IAM access keys
3. Enable Bedrock service

## 📡 API Endpoints

### Main API (Port 8000)
- **POST** `/upload-file/` - Generate Knowledge Objects from documents

### Chat API (Port 8001)  
- **POST** `/extract_text/` - Extract text using various OCR methods
- **POST** `/search_and_respond/` - Chat with documents using RAG

## 🎯 Use Cases

- **Document Analysis**: Extract insights from PDFs, DOCX, PPTX
- **Knowledge Management**: Generate structured knowledge articles
- **Visual Understanding**: Analyze images and charts with AI
- **Interactive Chat**: Q&A with document content
- **Multi-format Processing**: Handle text, images, and mixed content

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Frontend**: Streamlit
- **AI/ML**: LangChain, Transformers, PyTorch
- **OCR**: Tesseract, Florence-2, Cloud APIs
- **Search**: Qdrant, BM25S, Embeddings
- **Models**: Azure OpenAI, Google Gemini, Claude, Qwen

## 🐛 Troubleshooting

### Common Issues
1. **Connection Errors**: Ensure all servers are running on correct ports
2. **API Key Errors**: Verify keys in `config.properties`
3. **Import Errors**: Check virtual environment and dependencies
4. **Tesseract Errors**: Verify installation and path configuration

See **[COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)** for detailed troubleshooting.

## 📁 Project Structure

```
project-root/
├── florence2/                    # Main application
│   ├── main.py                   # FastAPI server (8000)
│   ├── chatapi.py                # FastAPI server (8001)  
│   ├── mainpage.py               # Streamlit main page
│   ├── config.properties         # API configuration
│   ├── pages/                    # Streamlit pages
│   └── API/                      # Backend modules
├── venv/                         # Virtual environment
├── requirements.txt              # Dependencies
├── COMPLETE_DOCUMENTATION.md     # Full documentation
└── README.md                     # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft Florence-2** for vision capabilities
- **OpenAI** for language models
- **Google** for Gemini models
- **Anthropic** for Claude
- **Tesseract** for OCR functionality
- **LangChain** for RAG framework

## 📞 Support

- 📖 Check [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md) for detailed guides
- 🐛 Report issues on GitHub
- 💬 Join discussions in the repository

---

<p align="center">
  <strong>🌟 Star this repository if you find it useful! 🌟</strong>
</p>
