#!/usr/bin/env python3
"""
Script to start both FastAPI servers in the background
"""
import subprocess
import sys
import time
import os

def start_server(script_name, port):
    """Start a FastAPI server in background"""
    try:
        print(f"Starting {script_name} on port {port}...")
        
        # Change to florence2 directory
        os.chdir('florence2')
        
        # Start the server
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"✅ {script_name} started with PID: {process.pid}")
        return process
        
    except Exception as e:
        print(f"❌ Failed to start {script_name}: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting FastAPI servers...")
    
    # Start main API server (port 8000)
    main_process = start_server("main.py", 8000)
    
    if main_process:
        print("🌟 FastAPI server started successfully!")
        print("📡 Main API server running on: http://localhost:8000")
        print("📋 Available endpoints:")
        print("   - POST /upload-file/ (for Knowledge Object generation)")
        print("\n🔧 To also start the chat API server (port 8001), run:")
        print("   python florence2/chatapi.py")
        print("\n🔗 Now you can use the Streamlit app without connection errors!")
    else:
        print("❌ Failed to start servers")
        sys.exit(1)
