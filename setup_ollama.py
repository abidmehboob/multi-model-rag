#!/usr/bin/env python3
"""
Setup script for RAG Application with Ollama
This script helps set up the environment and install Ollama if needed.
"""

import subprocess
import sys
import os
import platform
import requests

def run_command(command, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e}")
        if check:
            raise
        return None

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = run_command("ollama --version", check=False)
        if result:
            print(f"✅ Ollama is installed: {result}")
            return True
    except:
        pass
    
    print("❌ Ollama is not installed")
    return False

def install_ollama():
    """Install Ollama based on the platform"""
    system = platform.system().lower()
    
    print(f"🔧 Installing Ollama on {system}...")
    
    if system == "windows":
        print("📥 Please install Ollama manually from: https://ollama.com/download")
        print("After installation, restart this script.")
        return False
    
    elif system == "linux":
        try:
            run_command("curl -fsSL https://ollama.com/install.sh | sh")
            print("✅ Ollama installed successfully")
            return True
        except:
            print("❌ Failed to install Ollama automatically")
            return False
    
    elif system == "darwin":  # macOS
        print("📥 Please install Ollama manually from: https://ollama.com/download")
        print("Or use Homebrew: brew install ollama")
        return False
    
    else:
        print(f"❌ Unsupported platform: {system}")
        return False

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        # Try to connect to Ollama API
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
    except:
        pass
    
    print("❌ Ollama service is not running")
    return False

def start_ollama_service():
    """Start Ollama service"""
    print("🚀 Starting Ollama service...")
    
    system = platform.system().lower()
    
    if system == "windows":
        # On Windows, Ollama usually runs as a service
        print("💡 On Windows, start Ollama from the Start Menu or run 'ollama serve' in a separate terminal")
    else:
        try:
            # Start in background
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ Ollama service started in background")
            return True
        except:
            print("❌ Failed to start Ollama service")
            print("💡 Try running 'ollama serve' in a separate terminal")
            return False

def pull_model(model_name="llama3.1"):
    """Pull the specified model"""
    print(f"📥 Pulling model: {model_name}")
    
    try:
        run_command(f"ollama pull {model_name}")
        print(f"✅ Model {model_name} downloaded successfully")
        return True
    except:
        print(f"❌ Failed to pull model {model_name}")
        return False

def list_available_models():
    """List available Ollama models"""
    try:
        result = run_command("ollama list", check=False)
        if result:
            print("📋 Available models:")
            print(result)
            return True
    except:
        pass
    
    print("❌ Could not list models")
    return False

def main():
    """Main setup function"""
    print("🦙 RAG Application Setup with Ollama")
    print("=" * 50)
    
    # Check Python packages
    print("📦 Checking Python packages...")
    try:
        import langchain
        import sentence_transformers
        import faiss
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Run: py -m pip install -r requirements.txt")
        return False
    
    # Check Ollama installation
    if not check_ollama_installed():
        install_choice = input("Install Ollama? (y/n): ").strip().lower()
        if install_choice in ['y', 'yes']:
            if not install_ollama():
                return False
        else:
            print("❌ Ollama is required for this application")
            return False
    
    # Check Ollama service
    if not check_ollama_service():
        start_choice = input("Start Ollama service? (y/n): ").strip().lower()
        if start_choice in ['y', 'yes']:
            start_ollama_service()
            
            # Wait a moment for service to start
            import time
            time.sleep(3)
            
            if not check_ollama_service():
                print("❌ Could not start Ollama service")
                print("💡 Please start it manually: ollama serve")
                return False
    
    # List current models
    print("\n📋 Current models:")
    list_available_models()
    
    # Ask about pulling a model
    model_choice = input("\nPull llama3.1 model? (y/n): ").strip().lower()
    if model_choice in ['y', 'yes']:
        if not pull_model("llama3.1"):
            print("❌ Failed to pull model")
            return False
    
    print("\n✅ Setup complete!")
    print("🚀 You can now run the RAG application:")
    print("   python rag_app.py")
    print("   or")
    print("   python enhanced_rag_app.py")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
