#!/usr/bin/env python3
"""
Test the fixed multi-model parsing
"""

import requests
import json
import time

def test_multimodel():
    """Test the multi-model parsing functionality"""
    print("🧪 Testing Fixed Multi-Model Parsing")
    print("=" * 50)
    
    # Test question
    question = "Technical: How do car engines work? Creative: Write a story about cars"
    print(f"📝 Question: {question}")
    print()
    
    try:
        # Send request
        payload = {'question': question}
        response = requests.post(
            'http://localhost:5000/api/ask',
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        result = response.json()
        
        # Show parsing results
        print("✅ Request successful!")
        print(f"🎯 Parsing type: {result.get('prompt_type', 'Unknown')}")
        print(f"🧩 Sections found: {len(result.get('parsed_sections', []))}")
        print(f"✅ Successful models: {result.get('successful_models', 0)}")
        print(f"❌ Failed models: {result.get('failed_models', 0) if 'failed_models' in result else 0}")
        print()
        
        # Show results for each model
        results = result.get('results', [])
        if results:
            print("🔍 DETAILED RESULTS:")
            print("-" * 30)
            
            for i, res in enumerate(results, 1):
                model_type = res.get('model_type', 'Unknown').title()
                success = res.get('success', False)
                section_q = res.get('section_question', '')
                
                status = "✅ Success" if success else "❌ Error"
                print(f"{i}. {model_type} Model: {status}")
                print(f"   Question: \"{section_q}\"")
                
                if success:
                    answer = res.get('answer', '')
                    print(f"   Response: {answer[:100]}...")
                else:
                    error = res.get('error', 'Unknown error')
                    print(f"   Error: {error}")
                print()
        
        # Show if there are any remaining issues
        failed_models = result.get('failed_models', 0) if 'failed_models' in result else 0
        if failed_models > 0:
            print("⚠️ REMAINING ISSUES:")
            print("-" * 20)
            for res in results:
                if not res.get('success', True):
                    print(f"• {res.get('model_type', 'Unknown')}: {res.get('error', 'Unknown error')}")
        else:
            print("🎉 ALL MODELS WORKING CORRECTLY!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask app. Make sure it's running on http://localhost:5000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_multimodel()
