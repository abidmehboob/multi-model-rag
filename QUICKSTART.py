"""
QUICK START GUIDE
"""

print("🦙 RAG with LLaMA using Ollama - Quick Start")
print("=" * 50)
print()

print("📋 What was created:")
print("✅ Complete RAG application with LLaMA and Ollama")
print("✅ Basic and Enhanced versions")
print("✅ Configuration system")
print("✅ Setup and testing scripts")
print("✅ Sample document creator")
print("✅ Interactive launcher")
print()

print("🚀 Quick Start Steps:")
print("1. Install Python packages:")
print("   py -m pip install -r requirements.txt")
print()

print("2. Install Ollama:")
print("   - Windows: Download from https://ollama.com/download")
print("   - Linux: curl -fsSL https://ollama.com/install.sh | sh")
print()

print("3. Start Ollama and pull model:")
print("   ollama serve")
print("   ollama pull llama3.1")
print()

print("4. Run the application:")
print("   python run_app.py")
print()

print("🎯 Alternative: Use automated setup:")
print("   python setup_ollama.py")
print()

print("📄 Files Created:")
files = [
    ("rag_app.py", "Basic RAG implementation"),
    ("enhanced_rag_app.py", "Feature-rich RAG application"),
    ("config.py", "Configuration settings"),
    ("setup_ollama.py", "Automated setup script"),
    ("test_setup.py", "Validation tests"),
    ("create_sample_pdf.py", "Sample document creator"),
    ("run_app.py", "Interactive launcher"),
    ("requirements.txt", "Python dependencies"),
    ("README.md", "Detailed documentation")
]

for filename, description in files:
    print(f"  📄 {filename:<25} - {description}")

print()
print("💡 Features Implemented:")
features = [
    "📚 PDF document processing with PyPDFLoader",
    "🔍 FAISS vector storage for similarity search",
    "🤖 LLaMA model integration via Ollama",
    "💬 Interactive question-answering",
    "📊 Source document tracking",
    "💾 Persistent vector storage",
    "🔧 Configurable parameters",
    "📈 Multi-document support",
    "🧪 Comprehensive testing",
    "🚀 Easy setup and deployment"
]

for feature in features:
    print(f"  {feature}")

print()
print("🎉 Your RAG application is ready!")
print("📖 See README.md for detailed documentation")
print("🔗 Based on: https://medium.com/@danushidk507/rag-with-llama-using-ollama-a-deep-dive-into-retrieval-augmented-generation-c58b9a1cfcd3")
