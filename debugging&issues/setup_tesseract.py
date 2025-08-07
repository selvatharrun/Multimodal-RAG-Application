import requests
import zipfile
import os
import sys

def download_tesseract():
    """Download and extract portable Tesseract"""
    print("🔍 Setting up Tesseract OCR...")
    
    # Create tesseract directory
    tesseract_dir = os.path.join(os.getcwd(), "tesseract")
    os.makedirs(tesseract_dir, exist_ok=True)
    
    try:
        # Download portable Tesseract (using a GitHub release)
        print("📥 Downloading Tesseract...")
        url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.3.20231005/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
        
        # For now, let's use a simpler approach - download the installer and give instructions
        installer_path = os.path.join(tesseract_dir, "tesseract-installer.exe")
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(installer_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded installer to: {installer_path}")
            print("\n🔧 Installation Instructions:")
            print(f"1. Run: {installer_path}")
            print("2. Install to default location: C:\\Program Files\\Tesseract-OCR\\")
            print("3. The script will update config.properties automatically")
            
            return "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        else:
            print("❌ Failed to download Tesseract")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    tesseract_path = download_tesseract()
    if tesseract_path:
        print(f"\n📍 Expected Tesseract path: {tesseract_path}")
