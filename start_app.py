"""
Enhanced starter script for the RAG application
Uses 'py' command and improved error handling
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

def check_python_installation():
    """Check if Python is properly installed"""
    try:
        result = subprocess.run(['py', '--version'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"✅ Python found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Python not found with 'py' command")
            return False
    except Exception as e:
        print(f"❌ Error checking Python: {e}")
        return False

def install_requirements():
    """Install required packages"""
    requirements = [
        "flask",
        "flask-cors", 
        "requests",
        "scikit-learn",
        "numpy",
        "paramiko",
        "langchain",
        "langchain-community",
        "pypdf"
    ]
    
    print("🔧 Installing required packages...")
    
    for package in requirements:
        try:
            print(f"   Installing {package}...")
            result = subprocess.run([
                'py', '-m', 'pip', 'install', package, '--quiet'
            ], capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                print(f"   ✅ {package} installed successfully")
            else:
                print(f"   ⚠️ Warning installing {package}: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ Error installing {package}: {e}")

def check_ollama_connection():
    """Check if Ollama server is accessible"""
    try:
        import requests
        ollama_url = "http://20.185.83.16:8080/"
        headers = {"Authorization": "Bearer aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"}
        
        response = requests.get(f"{ollama_url}api/tags", headers=headers, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Ollama server accessible, found {len(models.get('models', []))} models")
            return True
        else:
            print(f"⚠️ Ollama server responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️ Could not connect to Ollama server: {e}")
        print("   The application will still work with fallback functionality")
        return False

def start_flask_app():
    """Start the Flask application"""
    print("\n🚀 Starting RAG Flask Application...")
    
    try:
        # Use 'py' command to start the Flask app
        process = subprocess.Popen([
            'py', 'app.py'
        ], shell=True, cwd=os.getcwd())
        
        print("✅ Flask application started!")
        print("🌐 Open your browser and go to:")
        print("   • Desktop: http://localhost:5000")
        print("   • Mobile: http://localhost:5000/mobile")
        print("\n📝 Press Ctrl+C to stop the application")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 Stopping application...")
        process.terminate()
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")

def main():
    """Main function"""
    print("🦙 RAG Application Starter")
    print("=" * 50)
    
    # Check Python installation
    if not check_python_installation():
        print("❌ Python installation issue. Please install Python or check PATH.")
        sys.exit(1)
    
    # Install requirements
    install_requirements()
    
    # Check Ollama connection
    check_ollama_connection()
    
    # Start Flask app
    start_flask_app()

if __name__ == "__main__":
    main()
