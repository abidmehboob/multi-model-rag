"""
Custom Ollama implementation with API key support for remote servers
"""

from typing import Any, Dict, List, Optional
import requests
import json

class CustomOllama:
    """
    Custom Ollama class that supports API key authentication for remote servers
    """
    
    def __init__(
        self,
        model: str = "gemma2",
        base_url: str = "http://20.185.83.16:8080/",
        api_key: str = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm",
        temperature: float = 0.2,
        **kwargs: Any,
    ):
        """
        Initialize CustomOllama with API key support
        
        Args:
            model: Name of the model to use
            base_url: Base URL of the Ollama API server
            api_key: API key for authentication
            temperature: Temperature for text generation
            **kwargs: Additional arguments
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.base_url = base_url.rstrip('/')
        
        # Prepare headers with API key
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Invoke the model with a prompt (LangChain-compatible interface)
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        return self._call(prompt, **kwargs)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Make a call to the Ollama API with custom authentication
        
        Args:
            prompt: The input prompt
            stop: Stop sequences
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        # Prepare the request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                **kwargs
            }
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            # Make API request
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self.headers,
                json=payload,
                timeout=120  # 2 minutes timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing Ollama response: {e}")
    
    def test_connection(self) -> bool:
        """
        Test connection to the Ollama server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/version",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models on the server
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            models = result.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
