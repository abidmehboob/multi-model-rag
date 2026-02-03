#!/usr/bin/env python3
"""
Quick model checker to see exact model names on your Ollama server
"""

import requests
import sys

# Your server configuration
OLLAMA_URL = "http://20.185.83.16:8080"
API_KEY = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"

def check_models():
    """Check available models and their exact names"""
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", headers=headers, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        models = result.get("models", [])
        
        print(f"🔍 Found {len(models)} models on server:")
        print("=" * 50)
        
        # Look for specific models your config is trying to use
        required_models = ["llama3.1", "llama3.2", "codellama", "qwen2.5", "gemma2"]
        
        available_models = []
        for model in models:
            name = model.get("name", "")
            available_models.append(name)
            size = model.get("size", 0)
            size_mb = round(size / (1024*1024), 1)
            
            # Highlight the models your app is looking for
            marker = "✅" if any(req in name.lower() for req in required_models) else "📦"
            print(f"{marker} {name} ({size_mb} MB)")
        
        print("\n" + "=" * 50)
        print("🎯 Model Mapping Analysis:")
        print("=" * 50)
        
        # Check each required model
        for req_model in required_models:
            matches = [name for name in available_models if req_model in name.lower()]
            if matches:
                print(f"✅ {req_model} → {matches}")
            else:
                print(f"❌ {req_model} → NOT FOUND")
                # Suggest alternatives
                if "llama" in req_model:
                    alternatives = [name for name in available_models if "llama" in name.lower()]
                    if alternatives:
                        print(f"   💡 Alternatives: {alternatives[:3]}")
        
        print("\n🔧 RECOMMENDED CONFIG UPDATES:")
        print("=" * 50)
        
        # Generate updated config based on available models
        for req_model in required_models:
            matches = [name for name in available_models if req_model in name.lower()]
            if matches:
                best_match = matches[0]  # Use first match
                if req_model != best_match:
                    print(f"   '{req_model}' → '{best_match}'")
        
        return available_models
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    try:
        check_models()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
