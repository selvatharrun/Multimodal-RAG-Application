@echo off
echo 🚀 Pushing Multimodal RAG Application to GitHub...
echo.

REM Check if we're in a git repository
if not exist .git (
    echo 📁 Initializing Git repository...
    git init
    git branch -M main
)

echo 📋 Adding files to Git...
git add .

echo 💬 Creating commit...
git commit -m "Complete Multimodal RAG Application with FastAPI and Streamlit - Initial release with comprehensive documentation, multiple AI models, advanced OCR, and intelligent search capabilities"

echo 🔗 Adding GitHub remote...
git remote remove origin 2>nul
git remote add origin https://github.com/selvatharrun/Multimodal-RAG-Application.git

echo 🚀 Pushing to GitHub...
git push -u origin main

echo.
echo ✅ Done! Your project should now be available at:
echo 🌐 https://github.com/selvatharrun/Multimodal-RAG-Application
echo.
echo 📋 Next steps:
echo 1. Update your config.properties with real API keys (use config.properties.template as reference)
echo 2. Install Tesseract OCR following TESSERACT_SETUP.md
echo 3. Review COMPLETE_DOCUMENTATION.md for setup instructions
echo.
pause
