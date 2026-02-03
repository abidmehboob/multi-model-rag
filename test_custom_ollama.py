"""
Test script for Custom Ollama Server Connection
Tests your specific Ollama server configuration
"""

import sys
from custom_ollama import CustomOllama

# Your Ollama server configuration
OLLAMA_MODEL = "gemma2"
OLLAMA_API_URL = "http://20.185.83.16:8080/"
OLLAMA_API_KEY = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"
DEFAULT_TEMPERATURE = 0.2

def test_custom_ollama():
    """Test the custom Ollama implementation"""
    print("🧪 Testing Custom Ollama Server Connection")
    print("=" * 50)
    print(f"🔗 Server URL: {OLLAMA_API_URL}")
    print(f"🤖 Model: {OLLAMA_MODEL}")
    print(f"🌡️  Temperature: {DEFAULT_TEMPERATURE}")
    print()
    
    try:
        # Initialize custom Ollama
        print("🚀 Initializing Custom Ollama...")
        ollama = CustomOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_API_URL,
            api_key=OLLAMA_API_KEY,
            temperature=DEFAULT_TEMPERATURE
        )
        print("✅ Custom Ollama initialized")
        
        # Test connection
        print("\n🔌 Testing server connection...")
        if ollama.test_connection():
            print("✅ Server connection successful")
        else:
            print("❌ Server connection failed")
            return False
        
        # Test model listing
        print("\n📋 Listing available models...")
        try:
            models = ollama.list_models()
            if models:
                print(f"✅ Found {len(models)} models:")
                for model in models:
                    print(f"   📦 {model}")
            else:
                print("⚠️  No models found or unable to list models")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
        
        # Test simple query
        print(f"\n🧪 Testing model '{OLLAMA_MODEL}' with simple query...")
        test_prompt = "Hello! Please respond with 'Test successful' to confirm you're working."
        
        response = ollama.invoke(test_prompt)
        print(f"📝 Model response: {response}")
        
        if response.strip():
            print("✅ Model is responding correctly")
        else:
            print("❌ Model returned empty response")
            return False
        
        # Test with a more complex query
        print(f"\n🧪 Testing with more complex query...")
        complex_prompt = "Explain what artificial intelligence is in one sentence."
        
        response = ollama.invoke(complex_prompt)
        print(f"📝 Complex response: {response[:200]}...")
        
        if len(response.strip()) > 10:
            print("✅ Model handles complex queries well")
        else:
            print("❌ Model response seems too short")
            return False
        
        # Test temperature variation
        print(f"\n🌡️  Testing temperature variation...")
        ollama_high_temp = CustomOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_API_URL,
            api_key=OLLAMA_API_KEY,
            temperature=0.8
        )
        
        creative_prompt = "Write a creative one-line story about a robot."
        
        response1 = ollama.invoke(creative_prompt)  # Low temperature
        response2 = ollama_high_temp.invoke(creative_prompt)  # High temperature
        
        print(f"📝 Low temp (0.2): {response1[:100]}...")
        print(f"📝 High temp (0.8): {response2[:100]}...")
        
        print("✅ Temperature variation test completed")
        
        print("\n🎉 All tests passed! Your Ollama server is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test function"""
    print("🦙 Custom Ollama Server Test Suite")
    print("Testing your remote Ollama configuration")
    print()
    
    # Test the custom implementation
    success = test_custom_ollama()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    if success:
        print("✅ All tests passed!")
        print("🚀 Your RAG application should work correctly.")
        print("\n💡 Next steps:")
        print("   1. Run: py simple_rag_app.py")
        print("   2. Or run: py run_app.py")
    else:
        print("❌ Some tests failed.")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if the server URL is accessible")
        print("   2. Verify the API key is correct")
        print("   3. Ensure the model 'gemma2' is available on the server")
        print("   4. Check network connectivity")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test cancelled by user")
        sys.exit(1)
