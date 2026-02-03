"""
Simplified RAG Application with Custom Ollama Integration
Works with your remote Ollama server configuration
"""

import os
import sys
from typing import List, Optional, Dict, Any
from custom_ollama import CustomOllama

# Try to import required packages
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    # Try new import first, fallback to old if needed
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ Using new langchain-huggingface package")
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("⚠️  Using deprecated HuggingFaceEmbeddings (consider upgrading to langchain-huggingface)")
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LangChain import error: {e}")
    print("💡 Install with: py -m pip install langchain langchain-community langchain-text-splitters")
    print("💡 For new embeddings: py -m pip install langchain-huggingface")
    LANGCHAIN_AVAILABLE = False

# Configuration
OLLAMA_MODEL = "gemma2"
OLLAMA_API_URL = "http://20.185.83.16:8080/"
OLLAMA_API_KEY = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"
DEFAULT_TEMPERATURE = 0.2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Valid Hugging Face model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 30
DEVICE = "cpu"
FAISS_INDEX_PATH = "faiss_index"

class SimpleRAGApplication:
    """
    Simplified RAG application using your custom Ollama server
    """
    
    def __init__(self):
        """Initialize the RAG application"""
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.embeddings = None
        
        print("🚀 Initializing Simple RAG Application with Custom Ollama...")
        self._setup_embeddings()
        self._setup_llm()
    
    def _setup_embeddings(self):
        """Setup the embedding model with fallback options"""
        if not LANGCHAIN_AVAILABLE:
            print("❌ LangChain not available. Cannot setup embeddings.")
            return
        
        # Try multiple embedding models in order of preference (corrected names without prefix)
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/distilbert-base-nli-mean-tokens"
        ]
        
        for model_name in embedding_models:
            print(f"📊 Trying embedding model: {model_name}")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": DEVICE},
                    encode_kwargs={'normalize_embeddings': False}
                )
                
                # Test the embedding
                print(f"🧪 Testing embedding model: {model_name}")
                test_embedding = self.embeddings.embed_query("Hello world")
                if test_embedding and len(test_embedding) > 0:
                    print(f"✅ Embedding model loaded successfully: {model_name}")
                    print(f"   Embedding dimension: {len(test_embedding)}")
                    # Update the global variable to remember which model worked
                    global EMBEDDING_MODEL
                    EMBEDDING_MODEL = model_name
                    return
                    
            except Exception as e:
                print(f"⚠️  Failed to load {model_name}: {e}")
                continue
        
        # Final fallback - try to use sentence-transformers directly with simpler names
        fallback_models = [
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2"
        ]
        
        for model_name in fallback_models:
            try:
                print(f"🔧 Trying direct sentence-transformers with model: {model_name}")
                from sentence_transformers import SentenceTransformer
                
                # Try loading with just the model name
                model = SentenceTransformer(model_name)
                print("✅ Loaded SentenceTransformer directly")
                
                # Create a simple wrapper that mimics HuggingFaceEmbeddings
                class DirectSentenceTransformerEmbeddings:
                    def __init__(self, model):
                        self.model = model
                    
                    def embed_query(self, text):
                        return self.model.encode([text])[0].tolist()
                    
                    def embed_documents(self, texts):
                        return [self.model.encode([text])[0].tolist() for text in texts]
                
                self.embeddings = DirectSentenceTransformerEmbeddings(model)
                print(f"✅ Direct sentence-transformers embedding setup complete with {model_name}")
                return
                
            except Exception as e:
                print(f"⚠️ Direct sentence-transformers with {model_name} failed: {e}")
                continue
        
        # If all models fail, raise an error
        raise Exception("❌ Could not load any embedding model. Please check internet connection or try: py -m pip install torch sentence-transformers transformers")
    
    def _setup_llm(self):
        """Setup the custom Ollama model"""
        print(f"🤖 Setting up Ollama model: {OLLAMA_MODEL}")
        print(f"🔗 Server: {OLLAMA_API_URL}")
        
        try:
            self.llm = CustomOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_API_URL,
                api_key=OLLAMA_API_KEY,
                temperature=DEFAULT_TEMPERATURE
            )
            
            # Test connection
            print("🔌 Testing connection to Ollama server...")
            if not self.llm.test_connection():
                raise Exception("Cannot connect to Ollama server")
            
            print("✅ Connection successful")
            
            # Test model
            print("🧪 Testing model response...")
            test_response = self.llm.invoke("Hello! Please respond with 'Connection test successful'")
            print(f"📝 Model response: {test_response[:100]}...")
            
        except Exception as e:
            print(f"❌ Error setting up Ollama model: {e}")
            print("\n📋 Check Configuration:")
            print(f"   Server URL: {OLLAMA_API_URL}")
            print(f"   Model: {OLLAMA_MODEL}")
            print("   API Key: [CONFIGURED]")
            raise
    
    def load_pdf_document(self, pdf_path: str):
        """
        Load and process a PDF document
        
        Args:
            pdf_path (str): Path to the PDF document
        """
        if not LANGCHAIN_AVAILABLE:
            print("❌ LangChain not available. Cannot load documents.")
            return None
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"📄 Loading PDF document: {pdf_path}")
        
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
            print(f"✂️ Split into {len(docs)} chunks")
            
            return docs
            
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise
    
    def create_vectorstore(self, docs):
        """
        Create FAISS vector store from documents
        
        Args:
            docs: Document chunks to vectorize
        """
        if not LANGCHAIN_AVAILABLE:
            print("❌ LangChain not available. Cannot create vectorstore.")
            return
        
        print("🔍 Creating vector embeddings...")
        
        try:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Save vectorstore
            print(f"💾 Saving vectorstore to '{FAISS_INDEX_PATH}'...")
            self.vectorstore.save_local(FAISS_INDEX_PATH)
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            print("✅ Vectorstore created and retriever configured")
            
        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            raise
    
    def load_existing_vectorstore(self):
        """Load existing FAISS vectorstore"""
        if not LANGCHAIN_AVAILABLE:
            print("❌ LangChain not available. Cannot load vectorstore.")
            return
        
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"Vectorstore not found: {FAISS_INDEX_PATH}")
        
        print(f"📂 Loading existing vectorstore from: {FAISS_INDEX_PATH}")
        
        try:
            self.vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            print("✅ Vectorstore loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            raise
    
    def query_documents(self, question: str) -> Dict[str, Any]:
        """
        Query the documents with a question
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict containing answer and source information
        """
        if not self.llm:
            raise ValueError("LLM not initialized")
        
        if not self.retriever:
            raise ValueError("Retriever not available. Load documents first.")
        
        print(f"\n❓ Question: {question}")
        print("🔍 Searching for relevant documents...")
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(question)
            print(f"📚 Found {len(docs)} relevant documents")
            
            if not docs:
                return {"answer": "No relevant documents found.", "sources": []}
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer: """
            
            # Get answer from LLM
            print("🤖 Generating answer...")
            answer = self.llm.invoke(prompt)
            
            # Prepare source information
            sources = []
            for i, doc in enumerate(docs):
                source_info = {
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "content_preview": doc.page_content[:100] + "..."
                }
                sources.append(source_info)
            
            result = {
                "answer": answer,
                "sources": sources
            }
            
            print(f"💡 Answer: {answer}")
            print(f"\n📚 Sources ({len(sources)} documents):")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. Page {source['page']} from {source['source']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during query: {e}")
            raise
    
    def interactive_mode(self):
        """Start interactive question-answering mode"""
        print("\n" + "="*70)
        print("🎯 Interactive RAG Mode - Ask questions about your documents!")
        print("💬 Commands: 'exit', 'quit', 'q' to quit")
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
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def create_sample_documents(self):
        """Create sample documents for testing without requiring PDF files"""
        if not LANGCHAIN_AVAILABLE:
            print("❌ LangChain not available. Cannot create vectorstore.")
            return None
        
        print("📚 Creating sample documents about AI and Machine Learning...")
        
        try:
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain.schema import Document
            except ImportError:
                from langchain.docstore.document import Document
        
        # Sample documents about AI/ML
        sample_docs = [
            Document(
                page_content="Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
                metadata={"source": "ai_intro", "page": 1}
            ),
            Document(
                page_content="Machine Learning (ML) is a subset of AI that focuses on algorithms and statistical models that enable computers to improve their performance on a specific task through experience. ML systems learn from data without being explicitly programmed for every scenario.",
                metadata={"source": "ml_basics", "page": 1}
            ),
            Document(
                page_content="Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in image recognition, natural language processing, and speech recognition.",
                metadata={"source": "deep_learning", "page": 1}
            ),
            Document(
                page_content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. NLP techniques include sentiment analysis, machine translation, text summarization, and question answering systems.",
                metadata={"source": "nlp_overview", "page": 1}
            ),
            Document(
                page_content="Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models with external knowledge sources. RAG systems retrieve relevant information from a knowledge base and use it to generate more accurate and contextually relevant responses.",
                metadata={"source": "rag_explanation", "page": 1}
            )
        ]
        
        return sample_docs

    def demo_mode_with_sample_docs(self):
        """Run a demo with sample documents instead of PDF"""
        print("🎮 Demo Mode: Using sample AI/ML documents")
        
        try:
            # Create sample documents
            docs = self.create_sample_documents()
            if not docs:
                return
            
            # Create vectorstore
            self.create_vectorstore(docs)
            
            # Run some demo queries
            demo_questions = [
                "What is artificial intelligence?",
                "How does machine learning work?", 
                "What is RAG?",
                "Explain deep learning"
            ]
            
            print("\n🤖 Running demo queries...")
            for question in demo_questions:
                print(f"\n{'='*60}")
                result = self.query_documents(question)
                print(f"{'='*60}")
            
            # Start interactive mode
            print("\n🎯 Demo complete! Now you can ask your own questions:")
            self.interactive_mode()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")

def main():
    """Main function"""
    print("🦙 Simple RAG Application with Custom Ollama")
    print("=" * 60)
    
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain is required but not available.")
        print("💡 Install with: py -m pip install langchain langchain-community")
        return
    
    try:
        # Initialize application
        rag_app = SimpleRAGApplication()
        
        # Check for existing vectorstore
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"📂 Found existing vectorstore: {FAISS_INDEX_PATH}")
            use_existing = input("Use existing vectorstore? (y/n): ").strip().lower()
            
            if use_existing in ['y', 'yes']:
                rag_app.load_existing_vectorstore()
                rag_app.interactive_mode()
                return
        
        # Show menu options
        print("\n📋 Select mode:")
        print("1. Load PDF document")
        print("2. Demo mode with sample AI/ML documents")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "2":
            # Run demo mode
            docs = rag_app.create_sample_documents()
            if docs:
                rag_app.create_vectorstore(docs)
                print("\n🎯 Demo complete! Ask questions about AI, ML, or RAG:")
                rag_app.interactive_mode()
            return
        elif choice == "1":
            # Load new document
            pdf_path = input("Enter path to your PDF document: ").strip()
            
            if not pdf_path:
                print("❌ No PDF path provided.")
                return
            
            # Process document
            docs = rag_app.load_pdf_document(pdf_path)
            rag_app.create_vectorstore(docs)
        else:
            print("❌ Invalid choice. Please select 1 or 2.")
            return
        
        # Start interactive mode
        rag_app.interactive_mode()
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
