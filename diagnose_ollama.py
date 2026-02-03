#!/usr/bin/env python3
"""
Ollama Server Diagnostic Tool
Helps identify and troubleshoot Ollama connectivity issues
"""

import requests
import time
import json
from datetime import datetime

# Configuration
OLLAMA_URL = "http://20.185.83.16:8080"
LOCAL_OLLAMA_URL = "http://localhost:11434"
API_KEY = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"

def test_connection(url, name, use_auth=False):
    """Test connection to Ollama server"""
    print(f"\n🔍 Testing {name} at {url}")
    print("-" * 50)
    
    # Prepare headers
    headers = {'Content-Type': 'application/json'}
    if use_auth and API_KEY:
        headers['Authorization'] = f'Bearer {API_KEY}'
        print(f"🔑 Using API key authentication")
    
    try:
        # Test basic connectivity
        print("1. Testing basic connectivity...")
        response = requests.get(f"{url}/api/version", headers=headers, timeout=10)
        print(f"   ✅ Server reachable - Status: {response.status_code}")
        
        if response.status_code == 200:
            version_data = response.json()
            print(f"   📋 Ollama Version: {version_data.get('version', 'Unknown')}")
        elif response.status_code == 403:
            print(f"   ⚠️ 403 Forbidden - Authentication may be required")
            if not use_auth:
                print(f"   🔄 Will retry with API key...")
                return False  # Signal to retry with auth
        
    except requests.exceptions.Timeout:
        print("   ❌ Connection timeout - Server not responding")
        return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection refused - Server not running or unreachable")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"   ⚠️ HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    try:
        # Test model list
        print("2. Testing model list...")
        response = requests.get(f"{url}/api/tags", headers=headers, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])
            print(f"   ✅ Found {len(models)} models")
            
            # List first 5 models
            for i, model in enumerate(models[:5]):
                name = model.get('name', 'Unknown')
                size = model.get('size', 0)
                size_mb = round(size / (1024*1024), 1) if size > 0 else 0
                print(f"   📦 {name} ({size_mb} MB)")
            
            if len(models) > 5:
                print(f"   📦 ... and {len(models) - 5} more models")
            
            return models
        elif response.status_code == 403:
            print(f"   ❌ 403 Forbidden - API key may be invalid or required")
            return False
        else:
            print(f"   ❌ Failed to get models - Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error getting models: {e}")
        return False

def test_model_generation(url, model_name, use_auth=False):
    """Test model generation"""
    print(f"\n🤖 Testing model generation with {model_name}")
    print("-" * 50)
    
    # Prepare headers
    headers = {'Content-Type': 'application/json'}
    if use_auth and API_KEY:
        headers['Authorization'] = f'Bearer {API_KEY}'
    
    test_prompt = "Hello, can you respond with just 'OK' to test the connection?"
    
    try:
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "stream": False
        }
        
        print(f"   📤 Sending test prompt to {model_name}...")
        start_time = time.time()
        
        response = requests.post(
            f"{url}/api/generate", 
            headers=headers,
            json=payload, 
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            print(f"   ✅ Model responded in {elapsed:.1f}s")
            print(f"   💬 Response: {answer[:100]}...")
            return True
        elif response.status_code == 403:
            print(f"   ❌ 403 Forbidden - Authentication required")
            return False
        else:
            print(f"   ❌ Generation failed - Status: {response.status_code}")
            print(f"   📄 Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ⏰ Timeout after 30 seconds")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("🔧 Ollama Server Diagnostic Tool")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test remote server first without authentication
    print(f"🔑 API Key available: {len(API_KEY)} characters")
    remote_models = test_connection(OLLAMA_URL, "Remote Ollama Server (No Auth)")
    
    # If failed, retry with authentication
    if not remote_models:
        print(f"\n🔄 Retrying with API key authentication...")
        remote_models = test_connection(OLLAMA_URL, "Remote Ollama Server (With Auth)", use_auth=True)
    
    # Test local server
    print("\n" + "=" * 60)
    local_models = test_connection(LOCAL_OLLAMA_URL, "Local Ollama Server")
    
    # If we have working servers, test model generation
    print("\n" + "=" * 60)
    
    if remote_models:
        # Try to test with llama3.1 or first available model
        test_models = ['llama3.1', 'gemma2', 'codellama']
        model_names = [m.get('name', '') for m in remote_models]
        
        for test_model in test_models:
            matching_models = [name for name in model_names if test_model in name.lower()]
            if matching_models:
                test_model_generation(OLLAMA_URL, matching_models[0], use_auth=True)
                break
        else:
            if model_names:
                test_model_generation(OLLAMA_URL, model_names[0], use_auth=True)
    
    if local_models:
        test_models = ['llama3.1', 'gemma2', 'codellama']
        model_names = [m.get('name', '') for m in local_models]
        
        for test_model in test_models:
            matching_models = [name for name in model_names if test_model in name.lower()]
            if matching_models:
                test_model_generation(LOCAL_OLLAMA_URL, matching_models[0])
                break
        else:
            if model_names:
                test_model_generation(LOCAL_OLLAMA_URL, model_names[0])
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if remote_models:
        print("✅ Remote server (20.185.83.16:8080) is working with API key")
        print(f"   📦 Available models: {len(remote_models)}")
        print(f"   🔑 Authentication: Required")
    else:
        print("❌ Remote server (20.185.83.16:8080) has issues")
        print("   🔧 Possible causes:")
        print("     • Invalid API key")
        print("     • Server configuration issues")
        print("     • Network connectivity problems")
        print("     • API key format incorrect")
    
    if local_models:
        print("✅ Local server (localhost:11434) is working")
        print(f"   📦 Available models: {len(local_models)}")
        print("   💡 Suggestion: Switch to local server for better performance")
    else:
        print("❌ Local server (localhost:11434) not available")
        print("   🔧 Install local Ollama:")
        print("     • Download from: https://ollama.ai")
        print("     • Run: ollama pull llama3.1")
        print("     • Run: ollama pull gemma2")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if remote_models:
        print("  ✅ Your remote server is working! The API key is valid.")
        print("  🔧 Make sure your application uses the same authentication.")
    elif not remote_models and not local_models:
        print("  🔧 Set up local Ollama for reliable access:")
        print("     1. Download from https://ollama.ai")
        print("     2. Install and run: ollama serve")
        print("     3. Pull models: ollama pull llama3.1")
        print("     4. Update your config to use localhost:11434")

if __name__ == "__main__":
    main()
