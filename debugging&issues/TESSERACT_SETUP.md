# 🔧 Tesseract OCR Installation Guide

## Quick Installation Steps:

### Option 1: Using Windows Package Manager (Recommended)
```powershell
# Run in PowerShell as Administrator
winget install --id UB-Mannheim.TesseractOCR
```

### Option 2: Manual Download and Install
1. **Download Tesseract:**
   - Go to: https://github.com/UB-Mannheim/tesseract/releases
   - Download: `tesseract-ocr-w64-setup-5.3.3.20231005.exe` (or latest version)

2. **Install:**
   - Run the installer as Administrator
   - Install to default location: `C:\Program Files\Tesseract-OCR\`
   - ✅ Make sure to check "Add to PATH" during installation

3. **Verify Installation:**
   ```powershell
   tesseract --version
   ```

### Option 3: Using Chocolatey
```powershell
# If you have Chocolatey installed
choco install tesseract -y
```

## ✅ Configuration Updated

Your `config.properties` file has been updated with the correct path:
```
[pytesseract]
file_path=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## 🧪 Test Installation

After installing Tesseract, test it with:
```python
import pytesseract
from PIL import Image

# Set the path (if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Test with a simple image
print("Tesseract version:", pytesseract.get_tesseract_version())
```

## 🔍 Troubleshooting

If you get "TesseractNotFoundError":
1. Check if Tesseract is installed: `where tesseract`
2. Update the path in config.properties
3. Restart your Python kernel/application

## 📋 Next Steps

Once Tesseract is installed:
1. ✅ Tesseract path updated in config.properties 
2. 🔧 You'll handle the Google API key
3. 🚀 Your OCR functionality should work!

The main FastAPI server should now be able to use Tesseract for OCR processing.
