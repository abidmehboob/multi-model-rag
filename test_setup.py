"""
Test script to validate the RAG application setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    packages = [
        ("langchain", "LangChain framework"),
        ("langchain_community", "LangChain community components"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS vector search"),
        ("pypdf", "PDF processing"),
        ("transformers", "Hugging Face Transformers"),
        ("torch", "PyTorch")
    ]
    
    failed = []
    
    for package, description in packages:
        try:
            __import__(package)
            print(f"  ✅ {package} - {description}")
        except ImportError as e:
            print(f"  ❌ {package} - {description}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("💡 Run: py -m pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully")
    return True

def test_ollama_connection():
    """Test connection to Ollama service"""
    print("\n🔌 Testing Ollama connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        
        if response.status_code == 200:
            version_info = response.json()
            print(f"  ✅ Ollama service is running (version: {version_info.get('version', 'unknown')})")
            return True
        else:
            print(f"  ❌ Ollama service responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  ❌ Cannot connect to Ollama service")
        print("  💡 Start Ollama service: ollama serve")
        return False
    except Exception as e:
        print(f"  ❌ Error testing Ollama connection: {e}")
        return False

def test_embedding_model():
    """Test embedding model loading"""
    print("\n🔤 Testing embedding model...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Test embedding
        test_text = ["Hello world", "This is a test"]
        vectors = embeddings.embed_documents(test_text)
        
        print(f"  ✅ Embedding model loaded successfully")
        print(f"  📊 Vector dimension: {len(vectors[0])}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading embedding model: {e}")
        return False

def test_ollama_model():
    """Test Ollama model functionality"""
    print("\n🤖 Testing Ollama model...")
    
    try:
        from langchain_community.llms import Ollama
        
        llm = Ollama(model="llama3.1")
        response = llm.invoke("Hello, respond with 'Test successful'")
        
        print(f"  ✅ Ollama model responded: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing Ollama model: {e}")
        print(f"  💡 Make sure the model is available: ollama pull llama3.1")
        return False

def test_faiss_functionality():
    """Test FAISS vector store functionality"""
    print("\n🔍 Testing FAISS functionality...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document
        
        # Create simple test documents
        docs = [
            Document(page_content="The sky is blue", metadata={"source": "test1"}),
            Document(page_content="Grass is green", metadata={"source": "test2"}),
            Document(page_content="The ocean is deep", metadata={"source": "test3"})
        ]
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Test search
        results = vectorstore.similarity_search("What color is the sky?", k=1)
        
        print(f"  ✅ FAISS vector store created successfully")
        print(f"  🔍 Search test result: {results[0].page_content}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing FAISS: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAG Application Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Ollama Connection", test_ollama_connection),
        ("Embedding Model", test_embedding_model),
        ("Ollama Model", test_ollama_model),
        ("FAISS Functionality", test_faiss_functionality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG application is ready to use.")
        print("🚀 You can now run:")
        print("   python rag_app.py")
        print("   python enhanced_rag_app.py")
    else:
        print("⚠️  Some tests failed. Please check the setup instructions.")
        print("📋 Setup checklist:")
        print("   1. Install Python packages: py -m pip install -r requirements.txt")
        print("   2. Install Ollama: https://ollama.com/download")
        print("   3. Start Ollama service: ollama serve")
        print("   4. Pull model: ollama pull llama3.1")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Tests cancelled by user")
        sys.exit(1)
