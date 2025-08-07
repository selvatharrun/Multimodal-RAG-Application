@echo off
echo 🚀 Starting Streamlit Multimodal RAG Chat App...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run Streamlit app
echo 🌟 Launching Streamlit app on http://localhost:8501
echo 📱 Your app will open in your default browser
echo.
streamlit run florence2/mainpage.py

pause
