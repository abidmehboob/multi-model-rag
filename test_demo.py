"""
Quick Test Script for RAG Application
This script tests the RAG application with demo mode to verify it works
"""

import sys
import os

def test_rag_demo():
    """Test the RAG application in demo mode"""
    print("🧪 Testing RAG Application in Demo Mode")
    print("=" * 50)
    
    # Import and run
    try:
        from simple_rag_app import SimpleRAGApplication
        
        # Initialize application
        app = SimpleRAGApplication()
        
        print("✅ Application initialized successfully")
        
        # Create sample documents
        docs = app.create_sample_documents()
        if not docs:
            print("❌ Failed to create sample documents")
            return False
        
        print(f"✅ Created {len(docs)} sample documents")
        
        # Test vectorstore creation
        app.create_vectorstore(docs)
        print("✅ Vectorstore created successfully")
        
        # Test a query
        test_question = "What is artificial intelligence?"
        print(f"\n🤖 Testing query: '{test_question}'")
        
        result = app.query_documents(test_question)
        print("✅ Query executed successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rag_demo()
    if success:
        print("\n🎉 Demo test completed successfully!")
        print("💡 You can now run: py simple_rag_app.py")
        print("   And select option 2 for demo mode")
    else:
        print("\n❌ Demo test failed. Check the errors above.")
        sys.exit(1)
