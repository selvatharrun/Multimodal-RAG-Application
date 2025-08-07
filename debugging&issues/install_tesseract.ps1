# PowerShell script to install Tesseract OCR
Write-Host "🔍 Installing Tesseract OCR..." -ForegroundColor Yellow

# Create a temporary directory
$tempDir = Join-Path $env:TEMP "tesseract_install"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    # Download Tesseract installer
    Write-Host "📥 Downloading Tesseract OCR installer..." -ForegroundColor Cyan
    $installerUrl = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
    $installerPath = Join-Path $tempDir "tesseract-installer.exe"
    
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
    
    # Install Tesseract
    Write-Host "🔧 Installing Tesseract OCR..." -ForegroundColor Cyan
    Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -NoNewWindow
    
    # Wait a moment for installation to complete
    Start-Sleep -Seconds 3
    
    # Check installation paths
    $possiblePaths = @(
        "C:\Program Files\Tesseract-OCR\tesseract.exe",
        "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe"
    )
    
    $tesseractPath = $null
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $tesseractPath = $path
            break
        }
    }
    
    if ($tesseractPath) {
        Write-Host "✅ Tesseract OCR installed successfully!" -ForegroundColor Green
        Write-Host "📍 Installation path: $tesseractPath" -ForegroundColor Green
        
        # Test Tesseract
        Write-Host "🧪 Testing Tesseract..." -ForegroundColor Cyan
        $version = & "$tesseractPath" --version 2>&1 | Select-Object -First 1
        Write-Host "📋 Version: $version" -ForegroundColor Green
        
        # Return the path for config update
        return $tesseractPath
    } else {
        Write-Host "❌ Tesseract installation failed or not found in expected locations" -ForegroundColor Red
        return $null
    }
    
} catch {
    Write-Host "❌ Error during installation: $($_.Exception.Message)" -ForegroundColor Red
    return $null
} finally {
    # Cleanup
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
