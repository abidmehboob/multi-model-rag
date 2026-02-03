"""
Simple test script to verify Ollama connection without LangChain dependencies
"""

from custom_ollama import CustomOllama

def test_ollama_only():
    """Test just the Ollama connection"""
    print("🧪 Testing Custom Ollama Connection Only")
    print("=" * 50)
    
    # Configuration
    OLLAMA_MODEL = "gemma2"
    OLLAMA_API_URL = "http://20.185.83.16:8080/"
    OLLAMA_API_KEY = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"
    DEFAULT_TEMPERATURE = 0.2
    
    print(f"🔗 Server: {OLLAMA_API_URL}")
    print(f"🤖 Model: {OLLAMA_MODEL}")
    print()
    
    try:
        # Initialize Ollama
        print("🚀 Initializing CustomOllama...")
        ollama = CustomOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_API_URL,
            api_key=OLLAMA_API_KEY,
            temperature=DEFAULT_TEMPERATURE
        )
        
        # Test connection
        print("🔌 Testing server connection...")
        if not ollama.test_connection():
            print("❌ Cannot connect to Ollama server")
            return False
        
        print("✅ Server connection successful")
        
        # Test basic query
        print("\n🧪 Testing basic query...")
        response = ollama.invoke("What is artificial intelligence?")
        print(f"📝 Response: {response[:200]}...")
        
        # Test interactive mode
        print("\n💬 Interactive Test Mode")
        print("Type 'exit' to quit")
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("🤖 Generating response...")
                answer = ollama.invoke(question)
                print(f"💡 Answer: {answer}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_ollama_only()
