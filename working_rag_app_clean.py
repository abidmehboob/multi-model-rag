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
        "model": "gemma2",
        "temperature": 0.2,
        "description": "General purpose AI assistant"
    },
    "technical": {
        "model": "llama3.1",  # Better for technical content
        "temperature": 0.1,
        "description": "Technical and scientific questions"
    },
    "creative": {
        "model": "gemma2",  # Good for creative tasks
        "temperature": 0.7,
        "description": "Creative writing and storytelling"
    },
    "code": {
        "model": "codellama",  # Specialized for code
        "temperature": 0.0,
        "description": "Code generation and programming"
    },
    "analysis": {
        "model": "llama3.1",  # Good for analytical thinking
        "temperature": 0.3,
        "description": "Data analysis and reasoning"
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
        
        success, output = self.execute_ssh_command("ollama list")
        
        if success:
            lines = output.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    if ':' in model_name:
                        model_name = model_name.split(':')[0]
                    self.available_models.add(model_name)
            
            print(f"✅ Found {len(self.available_models)} available models: {', '.join(self.available_models)}")
        else:
            print(f"⚠️ Could not fetch model list: {output}")
            # Add fallback models that are likely to be available
            self.available_models = {"gemma2", "llama3.1"}
    
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
        
        print("🚀 Initializing Simple RAG Application with Multi-Model Support...")
        
        # Initialize model manager for SSH-based model management
        self.model_manager = OllamaModelManager()
        
        self._setup_llm_models()
    
    def _setup_llm_models(self):
        """Setup multiple Ollama models for different prompt types"""
        print(f"🤖 Setting up multiple Ollama models...")
        
        for prompt_type, config in MULTI_MODEL_CONFIG.items():
            try:
                model_name = config['model']
                print(f"   📊 Checking {prompt_type} model: {model_name} (temp: {config['temperature']})")
                
                # Ensure model is available via SSH
                if self.model_manager.ensure_model_available(model_name):
                    # Initialize the model
                    self.llm_models[prompt_type] = CustomOllama(
                        model=model_name,
                        base_url=OLLAMA_API_URL,
                        api_key=OLLAMA_API_KEY,
                        temperature=config['temperature']
                    )
                    print(f"   ✅ {prompt_type} model initialized successfully")
                else:
                    print(f"   ❌ Failed to ensure {prompt_type} model is available")
                
            except Exception as e:
                print(f"   ⚠️ Failed to initialize {prompt_type} model: {e}")
                continue
        
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
        
        # Count keywords for each prompt type
        type_scores = {}
        for prompt_type, keywords in PROMPT_TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                type_scores[prompt_type] = score
        
        # Return the type with highest score, or default if none found
        if type_scores:
            detected_type = max(type_scores.keys(), key=lambda k: type_scores[k])
            print(f"🎯 Detected prompt type: {detected_type} (score: {type_scores[detected_type]})")
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
            config = MULTI_MODEL_CONFIG[prompt_type]
            print(f"🤖 Using {prompt_type} model: {config['description']}")
            return self.llm_models[prompt_type]
        
        # Fallback to default model
        print("🤖 Using default model as fallback")
        return self.llm_models.get('default', list(self.llm_models.values())[0])
    
    def ensure_model_for_prompt_type(self, prompt_type: str) -> bool:
        """Ensure the required model for prompt type is available, pull if necessary"""
        if prompt_type not in MULTI_MODEL_CONFIG:
            print(f"⚠️ Unknown prompt type: {prompt_type}")
            return False
            
        config = MULTI_MODEL_CONFIG[prompt_type]
        model_name = config['model']
        
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
    
    def query_documents(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with multi-model support"""
        print(f"\n❓ Question: {question}")
        
        # Detect prompt type and ensure model is available
        prompt_type = self.detect_prompt_type(question)
        
        # Ensure the required model is available (pull if necessary)
        if not self.ensure_model_for_prompt_type(prompt_type):
            print(f"⚠️ Could not set up {prompt_type} model, falling back to default")
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
            
            result = {
                "answer": formatted_answer,
                "sources": [],
                "source_type": "general_knowledge",
                "prompt_type": prompt_type,
                "model_config": MULTI_MODEL_CONFIG[prompt_type]
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
        
        result = {
            "answer": formatted_answer,
            "sources": sources,
            "source_type": "documents",
            "prompt_type": prompt_type,
            "model_config": MULTI_MODEL_CONFIG[prompt_type]
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
        print("📝 Commands: 'exit', 'quit', 'q' to quit")
        print("="*70)
        
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = self.query_documents(question)
                
                # Show source type and model info
                source_type = result.get('source_type', 'documents')
                prompt_type = result.get('prompt_type', 'default')
                
                if source_type == 'general_knowledge':
                    print(f"🌐 (Answer from Ollama's general knowledge using {prompt_type} model)")
                else:
                    print(f"📄 (Answer based on your documents using {prompt_type} model)")
                
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
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
