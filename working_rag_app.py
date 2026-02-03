"""
Simple Working RAG Application with Multi-Model Support
Uses basic TF-IDF embeddings to avoid dependency conflicts
"""

import os
import sys
import numpy as np
import subprocess
import paramiko
import json
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Try to import LangChain components
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ LangChain not fully available - some features may be limited")
    LANGCHAIN_AVAILABLE = False

from custom_ollama import CustomOllama

# Configuration
OLLAMA_MODEL = "gemma2"
OLLAMA_API_URL = "http://20.185.83.16:8080/"
OLLAMA_API_KEY = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"
DEFAULT_TEMPERATURE = 0.2
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 30
VECTORSTORE_PATH = "simple_vectorstore.pkl"

# SSH Configuration for Ollama server
SSH_CONFIG = {
    "host": "20.185.83.16",
    "port": 22,
    "user": "llama",
    "password": "Capadmin@024"
}

# Multi-model configuration with specific models for different types
MULTI_MODEL_CONFIG = {
    "default": {
        "model": "gemma2:latest",  # Updated to match server
        "temperature": 0.2,
        "description": "General purpose AI assistant"
    },
    "technical": {
        "model": "llama3.1:latest",  # Updated to match server
        "temperature": 0.1,
        "description": "Technical and scientific questions"
    },
    "creative": {
        "model": "gemma2:latest",  # Updated to match server
        "temperature": 0.7,
        "description": "Creative writing and storytelling"
    },
    "code": {
        "model": "codellama:latest",  # Updated to match server
        "temperature": 0.0,
        "description": "Code generation and programming"
    },
    "analysis": {
        "model": "llama3.1:latest",  # Updated to match server
        "temperature": 0.3,
        "description": "Data analysis and reasoning"
    },
    "chat": {
        "model": "gemma3:1b",  # Using lighter model for chat
        "temperature": 0.4,
        "description": "Quick conversational responses"
    },
    "reasoning": {
        "model": "qwen3:14b",  # Updated to match server
        "temperature": 0.1,
        "description": "Complex reasoning and problem solving"
    }
}

# Prompt type detection keywords
PROMPT_TYPE_KEYWORDS = {
    "technical": ["algorithm", "science", "research", "technical", "engineering", "mathematics", "physics", "chemistry", "biology", "architecture", "system", "design", "specification", "requirement", "documentation"],
    "creative": ["story", "creative", "write", "poem", "narrative", "fiction", "imagine", "create", "artistic", "brainstorm", "ideate"],
    "code": [
        # Programming languages
        "code", "programming", "python", "javascript", "java", "c++", "c#", "php", "ruby", "go", "rust", "kotlin", "swift", "typescript",
        # Code elements
        "function", "class", "method", "variable", "constant", "array", "list", "dictionary", "object", "module", "library", "package",
        # Programming concepts
        "debug", "syntax", "compile", "execute", "run", "implement", "develop", "build", "parse", "serialize", "deserialize",
        # Data formats
        "json", "xml", "csv", "yaml", "api", "rest", "graphql", "sql", "database", "query", "select", "insert", "update", "delete",
        # Control structures
        "loop", "iteration", "condition", "if", "else", "switch", "case", "while", "for", "foreach",
        # Web technologies
        "html", "css", "dom", "jquery", "react", "vue", "angular", "node", "express", "flask", "django",
        # Development tools
        "git", "github", "docker", "kubernetes", "deployment", "ci/cd", "testing", "unit test", "integration",
        # Patterns and practices
        "regex", "regular expression", "pattern", "validation", "authentication", "authorization", "security", "encryption"
    ],
    "analysis": ["analyze", "compare", "evaluate", "assess", "review", "examine", "study", "investigate", "data", "metrics", "performance", "optimization", "benchmark"]
}

class OllamaModelManager:
    """Manages Ollama models via SSH - pulls models if not available"""
    
    def __init__(self):
        self.ssh_config = SSH_CONFIG
        self.available_models = set()
        self.refresh_available_models()
    
    def execute_ssh_command(self, command: str) -> tuple[bool, str]:
        """Execute command on Ollama server via SSH"""
        try:
            # Try using paramiko if available
            try:
                import paramiko
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(
                    hostname=self.ssh_config["host"],
                    port=self.ssh_config["port"],
                    username=self.ssh_config["user"],
                    password=self.ssh_config["password"]
                )
                
                stdin, stdout, stderr = ssh.exec_command(command)
                output = stdout.read().decode('utf-8')
                error = stderr.read().decode('utf-8')
                
                ssh.close()
                
                if error and "Error" in error:
                    return False, error
                return True, output
                
            except ImportError:
                # Fallback to subprocess with ssh command
                print("⚠️ paramiko not available, using subprocess SSH")
                ssh_command = f'ssh {self.ssh_config["user"]}@{self.ssh_config["host"]} "{command}"'
                result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    return True, result.stdout
                else:
                    return False, result.stderr
                    
        except Exception as e:
            print(f"❌ SSH command failed: {e}")
            return False, str(e)
    
    def refresh_available_models(self):
        """Get list of available models from Ollama server"""
        print("🔍 Checking available models on Ollama server...")
        
        # Add common models as fallback
        fallback_models = {"llama3.2", "llama3.1", "codellama", "gemma2", "qwen2.5", "mistral", "phi3"}
        
        try:
            success, output = self.execute_ssh_command("ollama list")
            
            if success and output.strip():
                lines = output.strip().split('\n')
                if len(lines) > 1:  # Skip header if present
                    for line in lines[1:]:
                        if line.strip():
                            parts = line.split()
                            if parts:
                                model_name = parts[0]
                                if ':' in model_name:
                                    model_name = model_name.split(':')[0]
                                self.available_models.add(model_name)
                
                if self.available_models:
                    print(f"✅ Found {len(self.available_models)} available models: {', '.join(sorted(self.available_models))}")
                else:
                    print("⚠️ No models found in ollama list, using fallback models")
                    self.available_models = fallback_models
            else:
                print(f"⚠️ Could not fetch model list: {output}")
                print("🔄 Using fallback models")
                self.available_models = fallback_models
                
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            print("🔄 Using fallback models")
            self.available_models = fallback_models
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure model is available, pull if necessary"""
        base_model = model_name.split(':')[0]  # Remove tag if present
        
        if base_model in self.available_models:
            print(f"✅ Model {model_name} is already available")
            return True
        
        print(f"📥 Model {model_name} not found. Attempting to pull...")
        
        # Try to pull the model
        success, output = self.execute_ssh_command(f"ollama pull {model_name}")
        
        if success:
            print(f"✅ Successfully pulled model {model_name}")
            self.available_models.add(base_model)
            return True
        else:
            print(f"❌ Failed to pull model {model_name}: {output}")
            return False
    
    def get_model_info(self, model_name: str) -> dict:
        """Get information about a specific model"""
        success, output = self.execute_ssh_command(f"ollama show {model_name}")
        
        if success:
            return {"available": True, "info": output}
        else:
            return {"available": False, "error": output}

class SimpleDocument:
    """Simple document class"""
    def __init__(self, content: str, metadata: dict = None):
        self.page_content = content
        self.metadata = metadata or {}

class SimpleTFIDFVectorStore:
    """Simple TF-IDF based vector store"""
    
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.doc_vectors = None
        
    def add_documents(self, docs: List[SimpleDocument]):
        """Add documents to the vector store"""
        self.documents.extend(docs)
        
        # Extract text content
        texts = [doc.page_content for doc in self.documents]
        
        # Create TF-IDF vectors
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        print(f"✅ Added {len(docs)} documents to vector store")
        
    def similarity_search(self, query: str, k: int = 4) -> List[SimpleDocument]:
        """Search for similar documents"""
        if self.doc_vectors is None:
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top k most similar documents
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return [self.documents[i] for i in top_indices if similarities[i] > 0.1]
    
    def save(self, path: str):
        """Save vector store to file"""
        data = {
            'documents': self.documents,
            'vectorizer': self.vectorizer,
            'doc_vectors': self.doc_vectors
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Vector store saved to {path}")
    
    def load(self, path: str):
        """Load vector store from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.vectorizer = data['vectorizer']
        self.doc_vectors = data['doc_vectors']
        print(f"📂 Vector store loaded from {path}")

class SimpleRAG:
    """Simple RAG implementation with multi-model support"""
    
    def __init__(self):
        self.llm_models = {}  # Store multiple model instances
        self.vectorstore = SimpleTFIDFVectorStore()
        self.custom_prompt_types = {}  # Store user-defined prompt types
        
        print("🚀 Initializing Simple RAG Application with Multi-Model Support...")
        
        # Initialize model manager for SSH-based model management
        self.model_manager = OllamaModelManager()
        
        self._setup_llm_models()
        self._load_custom_types()
    
    def _setup_llm_models(self):
        """Setup multiple Ollama models for different prompt types"""
        print(f"🤖 Setting up multiple Ollama models...")
        
        # Track which models we successfully initialize
        successful_models = []
        failed_models = []
        
        for prompt_type, config in MULTI_MODEL_CONFIG.items():
            try:
                model_name = config['model']
                print(f"   📊 Checking {prompt_type} model: {model_name} (temp: {config['temperature']})")
                
                # Try to initialize the model without SSH check first (faster)
                try:
                    self.llm_models[prompt_type] = CustomOllama(
                        model=model_name,
                        base_url=OLLAMA_API_URL,
                        api_key=OLLAMA_API_KEY,
                        temperature=config['temperature']
                    )
                    print(f"   ✅ {prompt_type} model ({model_name}) initialized successfully")
                    successful_models.append(f"{prompt_type}:{model_name}")
                    
                except Exception as model_init_error:
                    print(f"   ⚠️ Failed to initialize {prompt_type} model directly: {model_init_error}")
                    
                    # Try to ensure model is available via SSH
                    if self.model_manager.ensure_model_available(model_name):
                        try:
                            self.llm_models[prompt_type] = CustomOllama(
                                model=model_name,
                                base_url=OLLAMA_API_URL,
                                api_key=OLLAMA_API_KEY,
                                temperature=config['temperature']
                            )
                            print(f"   ✅ {prompt_type} model initialized after SSH pull")
                            successful_models.append(f"{prompt_type}:{model_name}")
                        except Exception as retry_error:
                            print(f"   ❌ Failed to initialize {prompt_type} model after pull: {retry_error}")
                            failed_models.append(f"{prompt_type}:{model_name}")
                    else:
                        failed_models.append(f"{prompt_type}:{model_name}")
                
            except Exception as e:
                print(f"   ❌ Error setting up {prompt_type} model: {e}")
                failed_models.append(f"{prompt_type}:{config.get('model', 'unknown')}")
                continue
        
        # Ensure we have at least one working model
        if not successful_models:
            print("⚠️ No models initialized successfully. Attempting fallback setup...")
            self._setup_fallback_model()
        else:
            print(f"\n✅ Successfully initialized {len(successful_models)} models:")
            for model in successful_models:
                print(f"   • {model}")
            
            if failed_models:
                print(f"\n❌ Failed to initialize {len(failed_models)} models:")
                for model in failed_models:
                    print(f"   • {model}")
    
    def _setup_fallback_model(self):
        """Setup a single fallback model if all others fail"""
        fallback_models = ["llama3.2", "llama3.1", "gemma2", "mistral"]
        
        for model_name in fallback_models:
            try:
                print(f"🔄 Trying fallback model: {model_name}")
                fallback_llm = CustomOllama(
                    model=model_name,
                    base_url=OLLAMA_API_URL,
                    api_key=OLLAMA_API_KEY,
                    temperature=0.2
                )
                
                # If successful, use this model for all prompt types
                for prompt_type in MULTI_MODEL_CONFIG.keys():
                    self.llm_models[prompt_type] = fallback_llm
                
                print(f"✅ Fallback model {model_name} set up for all prompt types")
                return
                
            except Exception as e:
                print(f"❌ Fallback model {model_name} failed: {e}")
                continue
        
        print("❌ All fallback models failed. The application may not work properly.")
        raise Exception("Unable to initialize any Ollama models")
        
    def _load_custom_types(self):
        """Load custom prompt types from file if exists"""
        custom_types_file = "custom_prompt_types.json"
        if os.path.exists(custom_types_file):
            try:
                with open(custom_types_file, 'r', encoding='utf-8') as f:
                    self.custom_prompt_types = json.load(f)
                print(f"📋 Loaded {len(self.custom_prompt_types)} custom prompt types")
            except Exception as e:
                print(f"⚠️ Error loading custom types: {e}")
                self.custom_prompt_types = {}
    
    def _save_custom_types(self):
        """Save custom prompt types to file"""
        custom_types_file = "custom_prompt_types.json"
        try:
            with open(custom_types_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_prompt_types, f, indent=2)
            print(f"💾 Saved {len(self.custom_prompt_types)} custom prompt types")
        except Exception as e:
            print(f"❌ Error saving custom types: {e}")
    
    def add_custom_prompt_type(self, type_name: str, model_name: str, temperature: float, description: str, keywords: list):
        """Add a new custom prompt type and model configuration"""
        # Validate inputs
        if not type_name or not model_name:
            print("❌ Type name and model name are required")
            return False
            
        if temperature < 0 or temperature > 1:
            print("❌ Temperature must be between 0 and 1")
            return False
        
        # Clean model name (remove any whitespace)
        model_name = model_name.strip()
        
        print(f"🔍 Checking if model '{model_name}' is available...")
        
        # First, refresh the model list to get latest available models
        self.model_manager.refresh_available_models()
        
        # Check if model is already available locally
        base_model = model_name.split(':')[0]  # Remove tag if present
        if base_model not in self.model_manager.available_models:
            print(f"📥 Model '{model_name}' not found locally. Attempting to pull from Ollama repository...")
            
            # Try to pull the model
            success, output = self.model_manager.execute_ssh_command(f"ollama pull {model_name}")
            
            if success:
                print(f"✅ Successfully pulled model '{model_name}' from Ollama repository")
                # Add to available models
                self.model_manager.available_models.add(base_model)
                # Refresh the model list again to confirm
                self.model_manager.refresh_available_models()
            else:
                print(f"❌ Failed to pull model '{model_name}': {output}")
                print(f"💡 Make sure the model name is correct and available in Ollama repository")
                print(f"   You can check available models at: https://ollama.com/library")
                return False
        else:
            print(f"✅ Model '{model_name}' is already available locally")
        
        # Verify model availability one more time
        if not self.model_manager.ensure_model_available(model_name):
            print(f"❌ Model {model_name} is still not available after pull attempt")
            return False
        
        # Create custom type configuration
        custom_config = {
            "model": model_name,
            "temperature": temperature,
            "description": description,
            "keywords": keywords
        }
        
        # Add to custom types
        self.custom_prompt_types[type_name] = custom_config
        
        # Initialize the model
        print(f"🤖 Initializing model '{model_name}' for custom type '{type_name}'...")
        try:
            self.llm_models[type_name] = CustomOllama(
                model=model_name,
                base_url=OLLAMA_API_URL,
                api_key=OLLAMA_API_KEY,
                temperature=temperature
            )
            
            # Test the model with a simple query
            print("🧪 Testing model connection...")
            test_response = self.llm_models[type_name].invoke("Hello, please respond with 'Model test successful'")
            if test_response and "successful" in test_response.lower():
                print("✅ Model test successful!")
            else:
                print("⚠️ Model responded but test may have failed")
            
            # Save to file
            self._save_custom_types()
            
            print(f"✅ Added custom prompt type '{type_name}' with model '{model_name}'")
            print(f"   Description: {description}")
            print(f"   Keywords: {', '.join(keywords)}")
            print(f"   Temperature: {temperature}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize custom model: {e}")
            # Remove from custom types if model initialization failed
            if type_name in self.custom_prompt_types:
                del self.custom_prompt_types[type_name]
            print("💡 Possible issues:")
            print("   • Model may not be fully downloaded")
            print("   • Ollama server may be busy")
            print("   • Network connectivity issues")
            return False
    
    def list_prompt_types(self):
        """List all available prompt types (built-in and custom)"""
        print("\n📋 Available Prompt Types:")
        print("=" * 60)
        
        print("\n🔧 Built-in Types:")
        for ptype, config in MULTI_MODEL_CONFIG.items():
            status = "✅" if ptype in self.llm_models else "❌"
            print(f"  {status} {ptype}: {config['description']}")
            print(f"     Model: {config['model']} (temp: {config['temperature']})")
        
        if self.custom_prompt_types:
            print("\n🎨 Custom Types:")
            for ptype, config in self.custom_prompt_types.items():
                status = "✅" if ptype in self.llm_models else "❌"
                print(f"  {status} {ptype}: {config['description']}")
                print(f"     Model: {config['model']} (temp: {config['temperature']})")
                print(f"     Keywords: {', '.join(config['keywords'])}")
        else:
            print("\n🎨 No custom types defined")
    
    def remove_custom_prompt_type(self, type_name: str):
        """Remove a custom prompt type"""
        if type_name in self.custom_prompt_types:
            # Remove from custom types
            del self.custom_prompt_types[type_name]
            
            # Remove model if it exists
            if type_name in self.llm_models:
                del self.llm_models[type_name]
            
            # Save to file
            self._save_custom_types()
            
            print(f"✅ Removed custom prompt type '{type_name}'")
            return True
        else:
            print(f"❌ Custom prompt type '{type_name}' not found")
            return False
        
    def detect_prompt_type(self, question: str) -> str:
        """Detect the type of prompt based on keywords and content patterns"""
        question_lower = question.lower()
        
        # Check for code patterns first (higher priority)
        import re
        code_patterns = [
            r'def\s+\w+\s*\(',  # Python function definition
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+\w+',  # Import statements
            r'from\s+\w+\s+import',  # Python from import
            r'\w+\s*=\s*\w+\s*\(',  # Function assignment
            r'SELECT\s+.+FROM',  # SQL SELECT
            r'<[a-zA-Z]+[^>]*>',  # HTML tags
            r'\{[^}]*\}',  # JSON-like objects
            r'if\s*\([^)]+\)\s*\{',  # If statements with braces
            r'```[\w]*',  # Code blocks
            r'print\s*\(',  # Print statements
            r'console\.log\s*\(',  # Console log
            r'return\s+\w+',  # Return statements
        ]
        
        # Check for code patterns
        has_code_pattern = any(re.search(pattern, question, re.IGNORECASE) for pattern in code_patterns)
        
        if has_code_pattern:
            print("🎯 Detected code patterns in question - using code model")
            return "code"
        
        # Count keywords for each prompt type (built-in and custom)
        type_scores = {}
        
        # Check built-in types
        for prompt_type, keywords in PROMPT_TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                type_scores[prompt_type] = score
        
        # Check custom types
        for prompt_type, config in self.custom_prompt_types.items():
            keywords = config.get('keywords', [])
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                type_scores[prompt_type] = score
        
        # Return the type with highest score, or default if none found
        if type_scores:
            detected_type = max(type_scores.keys(), key=lambda k: type_scores[k])
            print(f"🎯 Detected prompt type: {detected_type} (score: {type_scores[detected_type]})")
            
            # Check if it's a custom type
            if detected_type in self.custom_prompt_types:
                print(f"🎨 Using custom prompt type: {detected_type}")
            
            return detected_type
        
        print("🎯 Using default prompt type")
        return "default"
    
    def format_code_response(self, response: str, prompt_type: str) -> str:
        """Format responses that might contain code"""
        if prompt_type != "code":
            return response
            
        # Add helpful formatting for code responses
        import re
        
        # Check if response already has code blocks
        has_code_blocks = bool(re.search(r'```[\w]*', response))
        
        if not has_code_blocks:
            # Try to detect and wrap code snippets
            lines = response.split('\n')
            formatted_lines = []
            in_code_block = False
            
            for line in lines:
                # Detect potential code lines
                code_indicators = [
                    r'^\s*(def|class|import|from|if|for|while|try|except)\s+',
                    r'^\s*\w+\s*=\s*\w+\s*\(',
                    r'^\s*(function|var|let|const)\s+',
                    r'^\s*console\.',
                    r'^\s*print\s*\(',
                    r'^\s*return\s+',
                    r'^\s*{\s*$',
                    r'^\s*}\s*$',
                ]
                
                is_code = any(re.match(pattern, line) for pattern in code_indicators)
                
                if is_code and not in_code_block:
                    formatted_lines.append('```python')
                    in_code_block = True
                elif not is_code and in_code_block and line.strip():
                    formatted_lines.append('```')
                    in_code_block = False
                
                formatted_lines.append(line)
            
            if in_code_block:
                formatted_lines.append('```')
            
            response = '\n'.join(formatted_lines)
        
        return response
    
    def get_llm_for_prompt_type(self, prompt_type: str) -> CustomOllama:
        """Get the appropriate LLM model for the prompt type"""
        if prompt_type in self.llm_models:
            # Check if it's a built-in or custom type
            if prompt_type in MULTI_MODEL_CONFIG:
                config = MULTI_MODEL_CONFIG[prompt_type]
                print(f"🤖 Using {prompt_type} model: {config['description']}")
            elif prompt_type in self.custom_prompt_types:
                config = self.custom_prompt_types[prompt_type]
                print(f"🎨 Using custom {prompt_type} model: {config['description']}")
            else:
                print(f"🤖 Using {prompt_type} model")
                
            return self.llm_models[prompt_type]
        
        # Fallback to default model
        print("🤖 Using default model as fallback")
        return self.llm_models.get('default', list(self.llm_models.values())[0])
    
    def ensure_model_for_prompt_type(self, prompt_type: str) -> bool:
        """Ensure the required model for prompt type is available, pull if necessary"""
        # Check if it's a built-in type
        if prompt_type in MULTI_MODEL_CONFIG:
            config = MULTI_MODEL_CONFIG[prompt_type]
            model_name = config['model']
        # Check if it's a custom type
        elif prompt_type in self.custom_prompt_types:
            config = self.custom_prompt_types[prompt_type]
            model_name = config['model']
        else:
            print(f"⚠️ Unknown prompt type: {prompt_type}")
            return False
        
        # Check if model is already initialized
        if prompt_type in self.llm_models:
            return True
        
        print(f"🔄 Model for {prompt_type} not available, attempting to set up...")
        
        # Use model manager to ensure model is available
        if self.model_manager.ensure_model_available(model_name):
            try:
                # Initialize the model
                self.llm_models[prompt_type] = CustomOllama(
                    model=model_name,
                    base_url=OLLAMA_API_URL,
                    api_key=OLLAMA_API_KEY,
                    temperature=config['temperature']
                )
                print(f"✅ {prompt_type} model ({model_name}) set up successfully")
                return True
                
            except Exception as e:
                print(f"❌ Failed to initialize {prompt_type} model: {e}")
                return False
        else:
            print(f"❌ Could not ensure {model_name} is available")
            return False
    
    def load_pdf_documents(self, pdf_paths: List[str]) -> List[SimpleDocument]:
        """Load PDF documents"""
        if not LANGCHAIN_AVAILABLE:
            print("❌ LangChain not available for PDF loading")
            return []
        
        all_docs = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠️ File not found: {pdf_path}")
                continue
            
            print(f"📄 Loading PDF: {pdf_path}")
            
            try:
                # Load PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                print(f"📖 Loaded {len(documents)} pages")
                
                # Split into chunks
                text_splitter = CharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separator="\n"
                )
                docs = text_splitter.split_documents(documents)
                
                # Convert to SimpleDocument
                for doc in docs:
                    simple_doc = SimpleDocument(
                        content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            'source_file': os.path.basename(pdf_path)
                        }
                    )
                    all_docs.append(simple_doc)
                
                print(f"✂️ Split into {len(docs)} chunks")
                
            except Exception as e:
                print(f"❌ Error loading {pdf_path}: {e}")
                continue
        
        return all_docs
    
    def load_text_documents(self, text_paths: List[str]) -> List[SimpleDocument]:
        """Load text documents"""
        all_docs = []
        
        for text_path in text_paths:
            if not os.path.exists(text_path):
                print(f"⚠️ File not found: {text_path}")
                continue
            
            print(f"📄 Loading text file: {text_path}")
            
            try:
                # Read text file
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks
                chunks = []
                if len(content) > CHUNK_SIZE:
                    for i in range(0, len(content), CHUNK_SIZE - CHUNK_OVERLAP):
                        chunk = content[i:i + CHUNK_SIZE]
                        if chunk.strip():
                            chunks.append(chunk)
                else:
                    chunks = [content]
                
                # Convert to SimpleDocument
                for i, chunk in enumerate(chunks):
                    simple_doc = SimpleDocument(
                        content=chunk,
                        metadata={
                            'source': os.path.basename(text_path),
                            'source_file': os.path.basename(text_path),
                            'chunk_id': i,
                            'topic': f"Text Document {i+1}"
                        }
                    )
                    all_docs.append(simple_doc)
                
                print(f"✂️ Split into {len(chunks)} chunks")
                
            except Exception as e:
                print(f"❌ Error loading {text_path}: {e}")
                continue
        
        return all_docs
    
    def create_sample_documents(self) -> List[SimpleDocument]:
        """Create sample documents for testing"""
        print("📚 Creating sample AI/ML documents...")
        
        sample_texts = [
            "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
            
            "Machine Learning (ML) is a subset of AI that focuses on algorithms and statistical models that enable computers to improve their performance on a specific task through experience. ML systems learn from data without being explicitly programmed for every scenario.",
            
            "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in image recognition, natural language processing, and speech recognition.",
            
            "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. NLP techniques include sentiment analysis, machine translation, text summarization, and question answering systems.",
            
            "Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge sources. RAG systems retrieve relevant information from a knowledge base and use it to generate more accurate and contextually relevant responses.",
            
            "Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using connectionist approaches to computation. Neural networks can learn and adapt through training on data.",
            
            "Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions or decisions. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values).",
            
            "Unsupervised learning algorithms find patterns in data without labeled examples. Common techniques include clustering (grouping similar data points) and dimensionality reduction (simplifying data while preserving important information)."
        ]
        
        docs = []
        for i, text in enumerate(sample_texts):
            doc = SimpleDocument(
                content=text,
                metadata={
                    'source': 'sample_ai_ml_docs',
                    'page': i+1,
                    'topic': f'AI/ML Topic {i+1}'
                }
            )
            docs.append(doc)
        
        print(f"✅ Created {len(docs)} sample documents")
        return docs
    
    def process_documents(self, docs: List[SimpleDocument]):
        """Process documents and create vector store"""
        print("🔍 Creating vector store...")
        self.vectorstore.add_documents(docs)
    
    def query_documents(self, question: str, force_prompt_type: str = None) -> Dict[str, Any]:
        """Query the RAG system with multi-model support"""
        print(f"\n❓ Question: {question}")
        
        # Detect prompt type and ensure model is available
        if force_prompt_type:
            print(f"🎯 Forcing prompt type: {force_prompt_type}")
            prompt_type = force_prompt_type
        else:
            prompt_type = self.detect_prompt_type(question)
        
        # Ensure the required model is available (pull if necessary)
        if not self.ensure_model_for_prompt_type(prompt_type):
            print(f"⚠️ Could not set up {prompt_type} model, falling back to default")
            if force_prompt_type:
                # If forcing a specific type that failed, return error
                return {
                    'answer': f"Error: Could not initialize {prompt_type} model",
                    'sources': [],
                    'source_type': 'error',
                    'prompt_type': prompt_type,
                    'model_config': {}
                }
            prompt_type = "default"
        
        llm = self.get_llm_for_prompt_type(prompt_type)
        
        print("🔍 Searching for relevant documents...")
        
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(question, k=4)
        print(f"📚 Found {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            # No relevant documents found - use Ollama model directly
            print("📝 No relevant documents found. Using Ollama model for general answer...")
            
            # Create a general prompt for the model
            general_prompt = f"""Please answer the following question based on your general knowledge:

Question: {question}

Answer: """
            
            # Get answer from LLM without context
            print("🤖 Generating general answer from Ollama model...")
            answer = llm.invoke(general_prompt)
            
            # Format code response if needed
            formatted_answer = self.format_code_response(answer, prompt_type)
            
            # Get model config (built-in or custom)
            if prompt_type in MULTI_MODEL_CONFIG:
                model_config = MULTI_MODEL_CONFIG[prompt_type]
            elif prompt_type in self.custom_prompt_types:
                model_config = self.custom_prompt_types[prompt_type]
            else:
                model_config = {"model": "unknown", "temperature": 0.2, "description": "Unknown model"}
            
            result = {
                "answer": formatted_answer,
                "sources": [],
                "source_type": "general_knowledge",
                "prompt_type": prompt_type,
                "model_config": model_config
            }
            
            print(f"💡 Answer (General Knowledge): {formatted_answer}")
            print(f"📚 Sources: General knowledge from Ollama model")
            print(f"🎯 Used {prompt_type} model configuration")
            
            return result
        
        # Create context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt with context
        prompt = f"""Based on the following context, please answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer: """
        
        # Get answer from LLM with context
        print("🤖 Generating answer based on documents...")
        answer = llm.invoke(prompt)
        
        # Format code response if needed
        formatted_answer = self.format_code_response(answer, prompt_type)
        
        # Prepare sources
        sources = []
        for doc in relevant_docs:
            source_info = {
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'Unknown'),
                'topic': doc.metadata.get('topic', 'Unknown'),
                'content_preview': doc.page_content[:100] + "...",
                'full_content': doc.page_content,  # Include full content for validation
                'source_file': doc.metadata.get('source_file', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 'Unknown')
            }
            sources.append(source_info)
        
        # Get model config (built-in or custom)
        if prompt_type in MULTI_MODEL_CONFIG:
            model_config = MULTI_MODEL_CONFIG[prompt_type]
        elif prompt_type in self.custom_prompt_types:
            model_config = self.custom_prompt_types[prompt_type]
        else:
            model_config = {"model": "unknown", "temperature": 0.2, "description": "Unknown model"}
        
        result = {
            "answer": formatted_answer,
            "sources": sources,
            "source_type": "documents",
            "prompt_type": prompt_type,
            "model_config": model_config
        }
        
        print(f"💡 Answer: {formatted_answer}")
        print(f"\n📚 Sources ({len(sources)} documents):")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source['topic']} (Page: {source['page']})")
            print(f"     File: {source.get('source_file', 'Unknown')}")
            print(f"     Preview: {source['content_preview']}")
        
        # Ask if user wants to see full source content
        print(f"\n💡 Full Answer: {formatted_answer}")
        print(f"🎯 Used {prompt_type} model configuration")
        print("\n" + "="*50)
        print("📖 SOURCE VALIDATION")
        print("="*50)
        
        for i, source in enumerate(sources, 1):
            print(f"\n📄 Source {i}: {source['topic']} (Page {source['page']})")
            print(f"📁 File: {source.get('source_file', 'Unknown')}")
            print("-" * 50)
            print(source.get('full_content', source['content_preview']))
            print("-" * 50)
        
        print(f"\n🎯 Used {prompt_type} model configuration")
        
        return result
    
    def interactive_mode(self):
        """Start interactive mode"""
        print("\n" + "="*70)
        print("🎯 Multi-Model RAG Application Ready!")
        print("💬 Ask questions about your documents or general topics")
        print("📄 Document-based answers will show sources")
        print("🌐 General questions will use Ollama's knowledge")
        print("🤖 Different models for different question types:")
        for ptype, config in MULTI_MODEL_CONFIG.items():
            print(f"   • {ptype}: {config['description']}")
        if self.custom_prompt_types:
            print("🎨 Custom types:")
            for ptype, config in self.custom_prompt_types.items():
                print(f"   • {ptype}: {config['description']}")
        print("\n📝 Commands:")
        print("   'exit', 'quit', 'q' - quit application")
        print("   'types' - list all prompt types")
        print("   'add type' - add new custom prompt type")
        print("   'modify type' - modify existing custom prompt type")
        print("   'remove type' - remove custom prompt type")
        print("   'map model' - map available Ollama model to type")
        print("   'models' - list available Ollama models")
        print("="*70)
        
        while True:
            try:
                user_input = input("\n🤔 Your question or command: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle management commands
                if user_input.lower() == 'types':
                    self.list_prompt_types()
                    continue
                elif user_input.lower() == 'add type':
                    self._interactive_add_type()
                    continue
                elif user_input.lower() == 'modify type':
                    self._interactive_modify_type()
                    continue
                elif user_input.lower() == 'remove type':
                    self._interactive_remove_type()
                    continue
                elif user_input.lower() == 'map model':
                    self._interactive_map_model()
                    continue
                elif user_input.lower() == 'models':
                    self._list_ollama_models()
                    continue
                
                # Process as a regular question
                result = self.query_documents(user_input)
                
                # Show source type and model info
                source_type = result.get('source_type', 'documents')
                prompt_type = result.get('prompt_type', 'default')
                
                if source_type == 'general_knowledge':
                    type_indicator = "🎨" if prompt_type in self.custom_prompt_types else "🌐"
                    print(f"{type_indicator} (Answer from Ollama's general knowledge using {prompt_type} model)")
                else:
                    type_indicator = "🎨" if prompt_type in self.custom_prompt_types else "📄"
                    print(f"{type_indicator} (Answer based on your documents using {prompt_type} model)")
                
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _interactive_add_type(self):
        """Interactive wizard to add custom prompt type"""
        print("\n🎨 Add Custom Prompt Type")
        print("=" * 40)
        
        try:
            # Get type name
            type_name = input("Enter prompt type name: ").strip().lower()
            if not type_name:
                print("❌ Type name cannot be empty")
                return
            
            if type_name in MULTI_MODEL_CONFIG or type_name in self.custom_prompt_types:
                print(f"❌ Type '{type_name}' already exists")
                return
            
            # Show available models and popular options
            print("\n📊 Available Models on Server:")
            self._list_ollama_models()
            
            print("\n🔥 Popular Models You Can Add:")
            popular_models = [
                ("llama3.2:1b", "Fast, lightweight model for quick responses"),
                ("llama3.2:3b", "Balanced speed and capability"),
                ("llama3.1:8b", "High capability for complex tasks"),
                ("mistral:7b", "Excellent for general use"),
                ("qwen2.5:7b", "Great for reasoning tasks"),
                ("phi3:mini", "Microsoft's efficient model"),
                ("gemma2:2b", "Google's compact model"),
                ("deepseek-coder", "Specialized for coding tasks"),
                ("nomic-embed-text", "For text embeddings"),
                ("wizard-vicuna", "Enhanced conversation model")
            ]
            
            print("\nModel Name               | Description")
            print("-" * 55)
            for model, desc in popular_models:
                print(f"{model:<23} | {desc}")
            
            print("\n💡 Tips:")
            print("   • Smaller models (1b, 3b) are faster")
            print("   • Larger models (8b, 70b) are more capable")
            print("   • Models will be automatically downloaded if not present")
            print("   • Check https://ollama.com/library for more models")
            
            # Get model name with validation
            while True:
                model_name = input("\nEnter Ollama model name (e.g., llama3.2:3b): ").strip()
                if not model_name:
                    print("❌ Model name cannot be empty")
                    continue
                
                # Basic validation of model name format
                if not model_name.replace(':', '').replace('-', '').replace('_', '').replace('.', '').replace('/', '').isalnum():
                    print("❌ Invalid model name format. Use alphanumeric characters, dots, colons, dashes, and slashes only.")
                    continue
                
                break
            
            # Get temperature with better explanation
            print("\n🌡️ Temperature Settings:")
            print("   0.0 = Very focused, deterministic (good for code, facts)")
            print("   0.2 = Slightly creative (good for general use)")
            print("   0.5 = Balanced creativity")
            print("   0.7 = More creative (good for writing)")
            print("   1.0 = Maximum creativity (very unpredictable)")
            
            try:
                temp_input = input("Enter temperature (0.0-1.0, default 0.2): ").strip()
                temperature = float(temp_input) if temp_input else 0.2
                if temperature < 0 or temperature > 1:
                    print("❌ Temperature must be between 0 and 1")
                    return
            except ValueError:
                print("❌ Invalid temperature value")
                return
            
            # Get description
            description = input("Enter description (what this type is for): ").strip()
            if not description:
                description = f"Custom {type_name} model using {model_name}"
            
            # Get keywords with examples
            print(f"\n🔑 Keywords help the system know when to use this model.")
            print(f"Examples: 'finance, banking, trading' or 'medical, health, diagnosis'")
            keywords_input = input("Enter keywords (comma-separated): ").strip()
            keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]
            
            if not keywords:
                print("❌ At least one keyword is required")
                return
            
            # Show summary before adding
            print(f"\n📋 Summary:")
            print(f"   Type Name: {type_name}")
            print(f"   Model: {model_name}")
            print(f"   Temperature: {temperature}")
            print(f"   Description: {description}")
            print(f"   Keywords: {', '.join(keywords)}")
            
            confirm = input(f"\n🤔 Add this custom type? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("❌ Cancelled")
                return
            
            # Add the custom type
            print(f"\n⏳ Adding custom prompt type '{type_name}'...")
            success = self.add_custom_prompt_type(type_name, model_name, temperature, description, keywords)
            
            if success:
                print(f"\n🎉 Successfully added custom prompt type '{type_name}'!")
                print("💡 You can now ask questions that include the specified keywords.")
                print("🧪 Try asking a question with your keywords to test it!")
            else:
                print(f"\n😞 Failed to add custom prompt type.")
            
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
    
    def _interactive_modify_type(self):
        """Interactive wizard to modify existing custom prompt type"""
        print("\n✏️ Modify Custom Prompt Type")
        print("=" * 40)
        
        if not self.custom_prompt_types:
            print("❌ No custom types to modify")
            return
        
        print("Available custom types:")
        type_list = list(self.custom_prompt_types.keys())
        for i, type_name in enumerate(type_list, 1):
            config = self.custom_prompt_types[type_name]
            print(f"  {i}. {type_name}")
            print(f"     Model: {config['model']} (temp: {config['temperature']})")
            print(f"     Description: {config['description']}")
            print(f"     Keywords: {', '.join(config['keywords'])}")
            print()
        
        try:
            choice = input("Enter type name or number to modify: ").strip()
            
            if choice.isdigit():
                # Modify by number
                idx = int(choice) - 1
                if 0 <= idx < len(type_list):
                    type_name = type_list[idx]
                else:
                    print("❌ Invalid number")
                    return
            else:
                # Modify by name
                type_name = choice.lower()
            
            if type_name not in self.custom_prompt_types:
                print(f"❌ Custom type '{type_name}' not found")
                return
            
            current_config = self.custom_prompt_types[type_name]
            print(f"\n📋 Current configuration for '{type_name}':")
            print(f"   Model: {current_config['model']}")
            print(f"   Temperature: {current_config['temperature']}")
            print(f"   Description: {current_config['description']}")
            print(f"   Keywords: {', '.join(current_config['keywords'])}")
            
            print(f"\n🔧 What would you like to modify?")
            print("1. Model name")
            print("2. Temperature")
            print("3. Description")
            print("4. Keywords")
            print("5. All fields")
            
            modify_choice = input("Enter your choice (1-5): ").strip()
            
            new_model = current_config['model']
            new_temp = current_config['temperature']
            new_desc = current_config['description']
            new_keywords = current_config['keywords']
            
            if modify_choice in ['1', '5']:
                # Show available models
                print(f"\n📊 Available Models on Server:")
                self._list_ollama_models()
                
                new_model_input = input(f"New model name (current: {current_config['model']}): ").strip()
                if new_model_input:
                    new_model = new_model_input
            
            if modify_choice in ['2', '5']:
                print(f"\n🌡️ Temperature Settings:")
                print("   0.0 = Very focused, deterministic")
                print("   0.2 = Slightly creative (default)")
                print("   0.7 = More creative")
                print("   1.0 = Maximum creativity")
                
                temp_input = input(f"New temperature (current: {current_config['temperature']}): ").strip()
                if temp_input:
                    try:
                        new_temp = float(temp_input)
                        if new_temp < 0 or new_temp > 1:
                            print("❌ Temperature must be between 0 and 1")
                            return
                    except ValueError:
                        print("❌ Invalid temperature value")
                        return
            
            if modify_choice in ['3', '5']:
                desc_input = input(f"New description (current: {current_config['description']}): ").strip()
                if desc_input:
                    new_desc = desc_input
            
            if modify_choice in ['4', '5']:
                keywords_input = input(f"New keywords (current: {', '.join(current_config['keywords'])}): ").strip()
                if keywords_input:
                    new_keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]
                    if not new_keywords:
                        print("❌ At least one keyword is required")
                        return
            
            # Show summary of changes
            print(f"\n📋 Summary of changes:")
            print(f"   Type Name: {type_name}")
            if new_model != current_config['model']:
                print(f"   Model: {current_config['model']} → {new_model}")
            if new_temp != current_config['temperature']:
                print(f"   Temperature: {current_config['temperature']} → {new_temp}")
            if new_desc != current_config['description']:
                print(f"   Description: {current_config['description']} → {new_desc}")
            if new_keywords != current_config['keywords']:
                print(f"   Keywords: {', '.join(current_config['keywords'])} → {', '.join(new_keywords)}")
            
            confirm = input(f"\n🤔 Apply these changes? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("❌ Cancelled")
                return
            
            # Remove old type and add new one
            print(f"\n⏳ Modifying custom prompt type '{type_name}'...")
            
            # Remove from model instances first
            if type_name in self.llm_models:
                del self.llm_models[type_name]
            
            # Add the modified custom type
            success = self.add_custom_prompt_type(type_name, new_model, new_temp, new_desc, new_keywords)
            
            if success:
                print(f"\n🎉 Successfully modified custom prompt type '{type_name}'!")
            else:
                print(f"\n😞 Failed to modify custom prompt type.")
                
        except (ValueError, KeyboardInterrupt):
            print("❌ Cancelled")
    
    def _interactive_map_model(self):
        """Interactive wizard to map available Ollama models to custom types"""
        print("\n🗺️ Map Available Model to Custom Type")
        print("=" * 50)
        
        # Refresh available models
        self.model_manager.refresh_available_models()
        
        if not self.model_manager.available_models:
            print("❌ No models available on Ollama server")
            return
        
        print("📊 Available Models on Server:")
        available_models = sorted(list(self.model_manager.available_models))
        for i, model in enumerate(available_models, 1):
            # Check if model is already mapped to a type
            mapped_types = []
            for type_name, config in self.custom_prompt_types.items():
                if config['model'].split(':')[0] == model:
                    mapped_types.append(type_name)
            
            # Check built-in types too
            for type_name, config in MULTI_MODEL_CONFIG.items():
                if config['model'].split(':')[0] == model:
                    mapped_types.append(f"{type_name}(built-in)")
            
            if mapped_types:
                print(f"  {i}. {model} → mapped to: {', '.join(mapped_types)}")
            else:
                print(f"  {i}. {model} → unmapped")
        
        try:
            model_choice = input(f"\nSelect model by number (1-{len(available_models)}): ").strip()
            
            if not model_choice.isdigit():
                print("❌ Invalid choice")
                return
                
            model_idx = int(model_choice) - 1
            if model_idx < 0 or model_idx >= len(available_models):
                print("❌ Invalid model number")
                return
            
            selected_model = available_models[model_idx]
            print(f"📦 Selected model: {selected_model}")
            
            # Check if model is already mapped
            existing_mappings = []
            for type_name, config in self.custom_prompt_types.items():
                if config['model'].split(':')[0] == selected_model:
                    existing_mappings.append(type_name)
            
            if existing_mappings:
                print(f"⚠️ Model '{selected_model}' is already mapped to custom types: {', '.join(existing_mappings)}")
                choice = input("Choose action:\n1. Create new mapping anyway\n2. Modify existing mapping\n3. Cancel\nYour choice (1-3): ").strip()
                
                if choice == "2":
                    # Modify existing mapping
                    if len(existing_mappings) == 1:
                        type_to_modify = existing_mappings[0]
                    else:
                        print("Multiple mappings found:")
                        for i, type_name in enumerate(existing_mappings, 1):
                            print(f"  {i}. {type_name}")
                        type_choice = input("Select type to modify: ").strip()
                        if type_choice.isdigit():
                            type_idx = int(type_choice) - 1
                            if 0 <= type_idx < len(existing_mappings):
                                type_to_modify = existing_mappings[type_idx]
                            else:
                                print("❌ Invalid choice")
                                return
                        else:
                            type_to_modify = type_choice.lower()
                    
                    if type_to_modify in self.custom_prompt_types:
                        print(f"🔧 Modifying existing type '{type_to_modify}'...")
                        # Call modify function for this specific type
                        self._modify_specific_type(type_to_modify, selected_model)
                    else:
                        print(f"❌ Type '{type_to_modify}' not found")
                    return
                    
                elif choice == "3":
                    print("❌ Cancelled")
                    return
                elif choice != "1":
                    print("❌ Invalid choice")
                    return
            
            # Create new mapping
            print(f"\n🎨 Creating new custom type for model '{selected_model}'")
            
            # Get type name
            type_name = input("Enter new prompt type name: ").strip().lower()
            if not type_name:
                print("❌ Type name cannot be empty")
                return
            
            if type_name in MULTI_MODEL_CONFIG or type_name in self.custom_prompt_types:
                print(f"❌ Type '{type_name}' already exists")
                return
            
            # Get temperature with smart defaults based on model type
            default_temp = 0.2
            if 'code' in selected_model.lower():
                default_temp = 0.0
                print(f"🤖 Detected code model - suggesting temperature 0.0 for deterministic code generation")
            elif any(x in selected_model.lower() for x in ['creative', 'story', 'write']):
                default_temp = 0.7
                print(f"✨ Detected creative model - suggesting temperature 0.7 for creative tasks")
            
            print(f"\n🌡️ Temperature (0.0-1.0, default {default_temp} for this model type): ")
            temp_input = input(f"Temperature [{default_temp}]: ").strip()
            try:
                temperature = float(temp_input) if temp_input else default_temp
                if temperature < 0 or temperature > 1:
                    print("❌ Temperature must be between 0 and 1")
                    return
            except ValueError:
                print("❌ Invalid temperature value")
                return
            
            # Get description with smart suggestions
            suggested_desc = f"Custom type using {selected_model} model"
            if 'code' in selected_model.lower():
                suggested_desc = f"Code generation and programming tasks using {selected_model}"
            elif any(x in selected_model.lower() for x in ['llama', 'mistral']):
                suggested_desc = f"General AI tasks using {selected_model}"
            elif 'qwen' in selected_model.lower():
                suggested_desc = f"Reasoning and analysis tasks using {selected_model}"
            
            description = input(f"Description [{suggested_desc}]: ").strip()
            if not description:
                description = suggested_desc
            
            # Get keywords with smart suggestions
            suggested_keywords = []
            if 'code' in selected_model.lower():
                suggested_keywords = ['code', 'programming', 'development', 'function', 'debug']
            elif 'qwen' in selected_model.lower():
                suggested_keywords = ['analysis', 'reasoning', 'logic', 'complex', 'think']
            elif any(x in selected_model.lower() for x in ['llama3.2:1b', '1b', '3b']):
                suggested_keywords = ['quick', 'fast', 'simple', 'brief']
            else:
                suggested_keywords = [type_name, selected_model.split(':')[0], 'general']
            
            print(f"💡 Suggested keywords based on model: {', '.join(suggested_keywords)}")
            keywords_input = input(f"Keywords (comma-separated) [{', '.join(suggested_keywords)}]: ").strip()
            
            if keywords_input:
                keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]
            else:
                keywords = suggested_keywords
            
            if not keywords:
                print("❌ At least one keyword is required")
                return
            
            # Show summary
            print(f"\n📋 Summary:")
            print(f"   Type Name: {type_name}")
            print(f"   Model: {selected_model}")
            print(f"   Temperature: {temperature}")
            print(f"   Description: {description}")
            print(f"   Keywords: {', '.join(keywords)}")
            
            confirm = input(f"\n🤔 Create this mapping? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("❌ Cancelled")
                return
            
            # Create the mapping
            print(f"\n⏳ Creating mapping for model '{selected_model}'...")
            success = self.add_custom_prompt_type(type_name, selected_model, temperature, description, keywords)
            
            if success:
                print(f"\n🎉 Successfully mapped model '{selected_model}' to type '{type_name}'!")
                print("💡 You can now ask questions with the specified keywords to use this model.")
            else:
                print(f"\n😞 Failed to create mapping.")
                
        except (ValueError, KeyboardInterrupt):
            print("❌ Cancelled")
    
    def _modify_specific_type(self, type_name: str, suggested_model: str = None):
        """Modify a specific type with optional model suggestion"""
        if type_name not in self.custom_prompt_types:
            print(f"❌ Type '{type_name}' not found")
            return
        
        current_config = self.custom_prompt_types[type_name]
        print(f"\n📋 Modifying '{type_name}':")
        print(f"   Current Model: {current_config['model']}")
        print(f"   Current Temperature: {current_config['temperature']}")
        print(f"   Current Description: {current_config['description']}")
        print(f"   Current Keywords: {', '.join(current_config['keywords'])}")
        
        # Model selection
        if suggested_model:
            model_input = input(f"New model [{suggested_model}]: ").strip()
            new_model = model_input if model_input else suggested_model
        else:
            model_input = input(f"New model (current: {current_config['model']}): ").strip()
            new_model = model_input if model_input else current_config['model']
        
        # Temperature
        temp_input = input(f"New temperature (current: {current_config['temperature']}): ").strip()
        if temp_input:
            try:
                new_temp = float(temp_input)
                if new_temp < 0 or new_temp > 1:
                    print("❌ Temperature must be between 0 and 1")
                    return
            except ValueError:
                print("❌ Invalid temperature value")
                return
        else:
            new_temp = current_config['temperature']
        
        # Description
        desc_input = input(f"New description (current: {current_config['description']}): ").strip()
        new_desc = desc_input if desc_input else current_config['description']
        
        # Keywords
        keywords_input = input(f"New keywords (current: {', '.join(current_config['keywords'])}): ").strip()
        if keywords_input:
            new_keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]
            if not new_keywords:
                print("❌ At least one keyword is required")
                return
        else:
            new_keywords = current_config['keywords']
        
        # Apply changes
        print(f"\n⏳ Updating type '{type_name}'...")
        
        # Remove old model instance
        if type_name in self.llm_models:
            del self.llm_models[type_name]
        
        # Add updated type
        success = self.add_custom_prompt_type(type_name, new_model, new_temp, new_desc, new_keywords)
        
        if success:
            print(f"✅ Successfully updated type '{type_name}'!")
        else:
            print(f"❌ Failed to update type '{type_name}'")

    def _list_ollama_models(self):
        """List available Ollama models with enhanced information"""
        print("\n🤖 Available Ollama Models:")
        print("=" * 50)
        
        # Refresh model list
        self.model_manager.refresh_available_models()
        
        if self.model_manager.available_models:
            print(f"📊 Found {len(self.model_manager.available_models)} models on server:")
            
            # Categorize models
            categories = {
                "LLaMA Models": [m for m in self.model_manager.available_models if 'llama' in m.lower()],
                "Code Models": [m for m in self.model_manager.available_models if any(x in m.lower() for x in ['code', 'coder'])],
                "Gemma Models": [m for m in self.model_manager.available_models if 'gemma' in m.lower()],
                "Qwen Models": [m for m in self.model_manager.available_models if 'qwen' in m.lower()],
                "Other Models": [m for m in self.model_manager.available_models if not any(x in m.lower() for x in ['llama', 'code', 'coder', 'gemma', 'qwen'])]
            }
            
            for category, models in categories.items():
                if models:
                    print(f"\n{category}:")
                    for model in sorted(models):
                        print(f"  • {model}")
        else:
            print("❌ No models found or unable to connect to Ollama server")
            print("\n🔧 Common Ollama models you can try:")
            common_models = [
                "llama3.2:1b", "llama3.2:3b", "llama3.2", 
                "llama3.1:8b", "llama3.1:70b", "llama3.1",
                "codellama:7b", "codellama:13b", "codellama",
                "gemma2:2b", "gemma2:9b", "gemma2:27b", "gemma2",
                "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5",
                "mistral:7b", "mistral", "phi3:mini", "phi3"
            ]
            
            for model in common_models:
                print(f"  • {model}")
        
        print(f"\n💡 Tips:")
        print(f"   • Use model names without size tags (e.g., 'llama3.2' instead of 'llama3.2:3b')")
        print(f"   • The system will attempt to pull models that aren't available")
        print(f"   • Smaller models (1b, 3b) are faster, larger models (70b) are more capable")
    
    def get_available_models(self):
        """Get list of available models for API"""
        self.model_manager.refresh_available_models()
        return {
            'models': list(self.model_manager.available_models),
            'total': len(self.model_manager.available_models),
            'working_models': list(self.llm_models.keys()),
            'custom_types': len(self.custom_prompt_types)
        }
    
    def save_vectorstore(self, path: str = VECTORSTORE_PATH):
        """Save vector store"""
        self.vectorstore.save(path)
    
    def load_vectorstore(self, path: str = VECTORSTORE_PATH):
        """Load vector store"""
        if os.path.exists(path):
            self.vectorstore.load(path)
            return True
        return False

def main():
    """Main function"""
    print("🦙 Multi-Model RAG Application (Smart Model Selection)")
    print("=" * 60)
    
    try:
        # Initialize RAG
        rag = SimpleRAG()
        
        # Check for existing vector store
        if os.path.exists(VECTORSTORE_PATH):
            print(f"📂 Found existing vector store: {VECTORSTORE_PATH}")
            use_existing = input("Use existing vector store? (y/n): ").strip().lower()
            
            if use_existing in ['y', 'yes']:
                rag.load_vectorstore()
                rag.interactive_mode()
                return
        
        # Show options
        print("\n📋 Select mode:")
        print("1. Load PDF documents")
        print("2. Demo mode with sample AI/ML documents")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "2":
            # Demo mode
            docs = rag.create_sample_documents()
            rag.process_documents(docs)
            rag.save_vectorstore()
            
            # Run demo queries - mix of different prompt types
            demo_questions = [
                "What is artificial intelligence?",  # Technical
                "How does machine learning work?",   # Technical
                "Write a creative story about a robot learning to paint",  # Creative
                "Create a Python function to calculate fibonacci numbers",  # Code
                "Analyze the differences between supervised and unsupervised learning",  # Analysis
                "What is the weather like today?", # General (no documents)
                "Tell me about quantum computing" # Technical (no documents)
            ]
            
            print("\n🤖 Running demo queries to showcase multi-model capabilities...")
            for question in demo_questions:
                result = rag.query_documents(question)
                source_type = result.get('source_type', 'documents')
                prompt_type = result.get('prompt_type', 'default')
                
                if source_type == 'general_knowledge':
                    print(f"🌐 (Used {prompt_type} model with general knowledge)")
                else:
                    print(f"📄 (Used {prompt_type} model with document knowledge)")
                print("\n" + "="*60)
            
            # Interactive mode
            print("\n🎯 Demo complete! Now you can ask your own questions:")
            print("💡 Try different types of questions to see different models in action!")
            rag.interactive_mode()
            
        elif choice == "1" and LANGCHAIN_AVAILABLE:
            # PDF mode
            print("\n📄 Enter PDF file paths (one per line, empty line to finish):")
            pdf_paths = []
            while True:
                path = input("PDF path: ").strip()
                if not path:
                    break
                pdf_paths.append(path)
            
            if pdf_paths:
                docs = rag.load_pdf_documents(pdf_paths)
                if docs:
                    rag.process_documents(docs)
                    rag.save_vectorstore()
                    rag.interactive_mode()
                else:
                    print("❌ No documents loaded")
            else:
                print("❌ No PDF paths provided")
                
        elif choice == "1" and not LANGCHAIN_AVAILABLE:
            print("❌ PDF loading requires LangChain. Please install: py -m pip install langchain langchain-community")
            
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
