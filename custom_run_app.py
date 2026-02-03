"""
Updated launcher for RAG application with Custom Ollama Server
Uses your remote Ollama server configuration
"""

import os
import sys
import subprocess

def print_banner():
    """Print application banner"""
    print("🦙" + "=" * 68)
    print("    RAG with Custom Ollama Server - Application Launcher")
    print("    Server: http://20.185.83.16:8080/")
    print("    Model: gemma2")
    print("=" * 71)

def check_requirements():
    """Check if requirements are installed"""
    try:
        import langchain
        import sentence_transformers
        import faiss
        import requests
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def test_ollama_server():
    """Test connection to your custom Ollama server"""
    print("🔌 Testing Ollama server connection...")
    try:
        result = subprocess.run([sys.executable, "test_custom_ollama.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ Ollama server test passed")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Ollama server test failed")
        print("🔍 Error details:", e.stderr if e.stderr else e.stdout)
        return False
    except FileNotFoundError:
        print("❌ test_custom_ollama.py not found")
        return False

def create_sample_document():
    """Create a sample document if needed"""
    # Check if we have any PDF files
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files:
        print("📄 No PDF files found. Creating sample document...")
        try:
            subprocess.run([sys.executable, "create_sample_pdf.py"], check=True)
        except:
            print("⚠️  Could not create sample PDF automatically")

def run_application(app_type="simple"):
    """Run the RAG application"""
    script_map = {
        "simple": "simple_rag_app.py",
        "enhanced": "enhanced_rag_app.py",
        "basic": "rag_app.py"
    }
    
    script = script_map.get(app_type, "simple_rag_app.py")
    
    if not os.path.exists(script):
        print(f"❌ {script} not found")
        return False
    
    print(f"🚀 Starting {app_type} RAG application...")
    print("📋 Configuration:")
    print("   Server: http://20.185.83.16:8080/")
    print("   Model: gemma2")
    print("   Temperature: 0.2")
    print()
    
    try:
        subprocess.run([sys.executable, script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Application failed")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check if we're in the right directory
    required_files = ["requirements.txt", "simple_rag_app.py", "custom_ollama.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("💡 Make sure you're in the correct directory")
        return False
    
    while True:
        print("\n🎯 What would you like to do?")
        print("1. 📦 Install Python requirements")
        print("2. 🔌 Test Ollama server connection")
        print("3. 📄 Create sample document")
        print("4. 🚀 Start Simple RAG application (Recommended)")
        print("5. ⭐ Start Enhanced RAG application")
        print("6. 📋 Show configuration info")
        print("7. 🆘 Show help")
        print("0. 🚪 Exit")
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        elif choice == "1":
            install_requirements()
        
        elif choice == "2":
            if not check_requirements():
                print("❌ Requirements not installed. Install them first (option 1).")
                continue
            test_ollama_server()
        
        elif choice == "3":
            create_sample_document()
        
        elif choice == "4":
            if not check_requirements():
                print("❌ Requirements not installed. Install them first (option 1).")
                continue
            run_application("simple")
        
        elif choice == "5":
            if not check_requirements():
                print("❌ Requirements not installed. Install them first (option 1).")
                continue
            run_application("enhanced")
        
        elif choice == "6":
            show_config_info()
        
        elif choice == "7":
            show_help()
        
        else:
            print("❌ Invalid choice. Please select 0-7.")

def show_config_info():
    """Show configuration information"""
    print("\n📋 Current Configuration")
    print("=" * 40)
    print(f"🔗 Ollama Server: http://20.185.83.16:8080/")
    print(f"🤖 Model: gemma2")
    print(f"🌡️  Temperature: 0.2")
    print(f"🔑 API Key: [CONFIGURED]")
    print(f"📊 Embedding Model: sentence-transformers/all-mpnet-base-v2")
    print(f"💻 Device: CPU")
    print(f"📚 Chunk Size: 1000")
    print(f"🔍 Search Results: 4 documents")

def show_help():
    """Show help information"""
    print("\n📚 RAG Application Help")
    print("=" * 40)
    print("""
🎯 Quick Start:
1. Install requirements (option 1)
2. Test server connection (option 2)  
3. Create or prepare a PDF document (option 3)
4. Start the Simple RAG application (option 4)

📋 Your Custom Configuration:
- Remote Ollama server at http://20.185.83.16:8080/
- Using gemma2 model with temperature 0.2
- API key authentication configured

🔧 Applications Available:
- Simple RAG: Straightforward implementation with your server
- Enhanced RAG: Feature-rich version (may have dependency issues)

💡 Troubleshooting:
- If connection fails, check server URL and API key
- If model errors occur, verify 'gemma2' is available on server
- For import errors, reinstall requirements

📁 Files Structure:
- simple_rag_app.py: Recommended application
- custom_ollama.py: Custom server implementation
- test_custom_ollama.py: Server connection tests
- config.py: Configuration settings

🔗 Your Server Details:
- URL: http://20.185.83.16:8080/
- Model: gemma2  
- Temperature: 0.2
- Authentication: API key configured
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Launcher stopped by user")
    except Exception as e:
        print(f"❌ Launcher error: {e}")
