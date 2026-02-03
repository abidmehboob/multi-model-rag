"""
Main launcher script for the RAG application
"""

import os
import sys
import subprocess

def print_banner():
    """Print application banner"""
    print("🦙" + "=" * 60)
    print("    RAG with LLaMA using Ollama - Application Launcher")
    print("=" * 63)

def check_requirements():
    """Check if requirements are installed"""
    try:
        import langchain
        import sentence_transformers
        import faiss
        return True
    except ImportError:
        return False

def run_setup():
    """Run the setup script"""
    print("🔧 Running setup script...")
    try:
        subprocess.run([sys.executable, "setup_ollama.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Setup failed")
        return False
    except FileNotFoundError:
        print("❌ setup_ollama.py not found")
        return False

def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    try:
        result = subprocess.run([sys.executable, "test_setup.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Tests failed")
        return False
    except FileNotFoundError:
        print("❌ test_setup.py not found")
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

def run_application(app_type="enhanced"):
    """Run the RAG application"""
    script_map = {
        "basic": "rag_app.py",
        "enhanced": "enhanced_rag_app.py"
    }
    
    script = script_map.get(app_type, "enhanced_rag_app.py")
    
    if not os.path.exists(script):
        print(f"❌ {script} not found")
        return False
    
    print(f"🚀 Starting {app_type} RAG application...")
    try:
        subprocess.run([sys.executable, script], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Application failed")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True

def main():
    """Main launcher function"""
    print_banner()
    
    # Check if we're in the right directory
    required_files = ["requirements.txt", "rag_app.py", "enhanced_rag_app.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        print("💡 Make sure you're in the correct directory")
        return False
    
    while True:
        print("\n🎯 What would you like to do?")
        print("1. 🔧 Run setup (install Ollama, pull models)")
        print("2. 🧪 Run tests (validate setup)")
        print("3. 📄 Create sample document")
        print("4. 🚀 Start Basic RAG application")
        print("5. ⭐ Start Enhanced RAG application")
        print("6. 📋 Show help")
        print("0. 🚪 Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        elif choice == "1":
            if not run_setup():
                print("⚠️  Setup incomplete. Some features may not work.")
        
        elif choice == "2":
            if not check_requirements():
                print("❌ Requirements not installed. Run setup first.")
                continue
            run_tests()
        
        elif choice == "3":
            create_sample_document()
        
        elif choice == "4":
            if not check_requirements():
                print("❌ Requirements not installed. Run setup first.")
                continue
            run_application("basic")
        
        elif choice == "5":
            if not check_requirements():
                print("❌ Requirements not installed. Run setup first.")
                continue
            run_application("enhanced")
        
        elif choice == "6":
            show_help()
        
        else:
            print("❌ Invalid choice. Please select 0-6.")

def show_help():
    """Show help information"""
    print("\n📚 RAG Application Help")
    print("=" * 40)
    print("""
🎯 Quick Start:
1. Run setup to install Ollama and models
2. Run tests to validate everything works
3. Create a sample document (or prepare your own PDF)
4. Start the enhanced application

📋 Prerequisites:
- Python 3.8+
- Internet connection (for downloading models)
- At least 8GB RAM (16GB recommended)

🔧 Manual Setup:
If automatic setup fails, you can:
1. Install Ollama from https://ollama.com/download
2. Run: ollama serve
3. Run: ollama pull llama3.1
4. Install Python packages: py -m pip install -r requirements.txt

💡 Tips:
- Use the Enhanced application for better features
- GPU acceleration requires CUDA setup
- Larger models provide better quality but need more memory

📁 File Structure:
- rag_app.py: Basic implementation
- enhanced_rag_app.py: Feature-rich version
- config.py: Configuration settings
- test_setup.py: Validation tests
- setup_ollama.py: Automated setup

🔗 Resources:
- Original article: https://medium.com/@danushidk507/rag-with-llama-using-ollama-a-deep-dive-into-retrieval-augmented-generation-c58b9a1cfcd3
- Ollama: https://ollama.com
- LangChain: https://langchain.readthedocs.io
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Launcher stopped by user")
    except Exception as e:
        print(f"❌ Launcher error: {e}")
