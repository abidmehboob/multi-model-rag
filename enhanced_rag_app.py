"""
Enhanced RAG Application with Configuration Support
"""

import os
import sys
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from custom_ollama import CustomOllama  # Use our custom implementation

# Import configuration
try:
    from config import *
except ImportError:
    # Fallback configuration if config.py is not available
    OLLAMA_MODEL = "gemma2"
    OLLAMA_API_URL = "http://20.185.83.16:8080/"
    OLLAMA_API_KEY = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"
    DEFAULT_TEMPERATURE = 0.2
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 30
    SEPARATOR = "\n"
    DEVICE = "cpu"
    FAISS_INDEX_PATH = "faiss_index"
    SEARCH_KWARGS = {"k": 4}
    CHAIN_TYPE = "stuff"

class EnhancedRAGApplication:
    def __init__(self, model_name: str = OLLAMA_MODEL):
        """
        Initialize the Enhanced RAG application
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        
        print("🚀 Initializing Enhanced RAG Application...")
        self._setup_embeddings()
        self._setup_llm()
    
    def _setup_embeddings(self):
        """Setup the embedding model"""
        print(f"📊 Setting up embedding model: {EMBEDDING_MODEL}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": DEVICE}
            )
            print(f"✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def _setup_llm(self):
        """Setup the LLaMA model using Custom Ollama"""
        print(f"🤖 Setting up model: {self.model_name}")
        try:
            self.llm = CustomOllama(
                model=self.model_name,
                base_url=OLLAMA_API_URL,
                api_key=OLLAMA_API_KEY,
                temperature=DEFAULT_TEMPERATURE
            )
            
            # Test the connection
            if not self.llm.test_connection():
                raise Exception("Cannot connect to Ollama server")
            
            # Test the model
            test_response = self.llm.invoke("Hello, are you working?")
            print(f"✅ Model loaded successfully")
            print(f"🧪 Test response: {test_response[:100]}...")
            
        except Exception as e:
            print(f"❌ Error setting up model: {e}")
            print("\n📋 Check Your Configuration:")
            print(f"   Server URL: {OLLAMA_API_URL}")
            print(f"   Model: {self.model_name}")
            print("   Make sure the server is running and the API key is correct")
            raise
    
    def load_documents(self, file_paths: List[str]) -> List:
        """
        Load and process multiple PDF documents
        
        Args:
            file_paths (List[str]): List of paths to PDF documents
            
        Returns:
            List: Processed document chunks
        """
        all_docs = []
        
        for pdf_path in file_paths:
            if not os.path.exists(pdf_path):
                print(f"⚠️ File not found: {pdf_path}")
                continue
            
            print(f"📄 Loading document: {pdf_path}")
            
            try:
                # Load the PDF document
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                print(f"📖 Loaded {len(documents)} pages from {os.path.basename(pdf_path)}")
                
                # Add source information to metadata
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_path)
                
                all_docs.extend(documents)
                
            except Exception as e:
                print(f"❌ Error loading {pdf_path}: {e}")
                continue
        
        if not all_docs:
            raise ValueError("No documents were successfully loaded")
        
        # Split documents into chunks
        print(f"✂️ Splitting {len(all_docs)} pages into chunks...")
        text_splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separator=SEPARATOR
        )
        docs = text_splitter.split_documents(all_docs)
        print(f"📚 Created {len(docs)} document chunks")
        
        return docs
    
    def create_vectorstore(self, docs: List):
        """
        Create FAISS vector store from document chunks
        
        Args:
            docs (List): Document chunks to vectorize
        """
        print("🔍 Creating vector embeddings and FAISS index...")
        
        try:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Save the vector store
            print(f"💾 Saving vector store to '{FAISS_INDEX_PATH}'...")
            self.vectorstore.save_local(FAISS_INDEX_PATH)
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs=SEARCH_KWARGS
            )
            print("✅ Vector store created and retriever configured")
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def load_existing_vectorstore(self, index_path: str = FAISS_INDEX_PATH):
        """
        Load an existing FAISS vector store
        
        Args:
            index_path (str): Path to the FAISS index directory
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        print(f"📂 Loading existing vector store from: {index_path}")
        
        try:
            self.vectorstore = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs=SEARCH_KWARGS
            )
            print("✅ Vector store loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            raise
    
    def setup_qa_chain(self):
        """Setup the RetrievalQA chain"""
        if not self.retriever:
            raise ValueError("Retriever not initialized. Load documents first.")
        
        if not self.llm:
            raise ValueError("LLM not initialized.")
        
        print("🔗 Setting up RetrievalQA chain...")
        
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=CHAIN_TYPE,
                retriever=self.retriever,
                return_source_documents=True
            )
            print("✅ QA chain setup complete")
            
        except Exception as e:
            print(f"❌ Error setting up QA chain: {e}")
            raise
    
    def query(self, question: str) -> dict:
        """
        Ask a question and get an answer based on the documents
        
        Args:
            question (str): The question to ask
            
        Returns:
            dict: The answer and source documents from the RAG system
        """
        if not self.qa_chain:
            raise ValueError("QA chain not setup. Call setup_qa_chain() first.")
        
        print(f"\n❓ Question: {question}")
        print("🔍 Searching for relevant information...")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            answer = result.get("result", "No answer found")
            sources = result.get("source_documents", [])
            
            print(f"💡 Answer: {answer}")
            
            if sources:
                print(f"\n📚 Sources ({len(sources)} documents):")
                for i, doc in enumerate(sources, 1):
                    source_file = doc.metadata.get('source_file', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    print(f"  {i}. {source_file} (Page: {page})")
            
            return {"answer": answer, "sources": sources}
            
        except Exception as e:
            print(f"❌ Error during query: {e}")
            raise
    
    def interactive_query_loop(self):
        """Start an interactive query loop"""
        if not self.qa_chain:
            print("🔧 Setting up QA chain...")
            self.setup_qa_chain()
        
        print("\n" + "="*70)
        print("🎯 RAG Application Ready!")
        print("💬 Type your questions below")
        print("📝 Commands: 'exit', 'quit', 'q' to quit")
        print("="*70)
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                result = self.query(query)
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the Enhanced RAG application"""
    print("🦙 RAG with LLaMA using Ollama - Enhanced Version")
    print("=" * 60)
    
    try:
        # Initialize the application
        rag_app = EnhancedRAGApplication()
        
        # Check if there's an existing vector store
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"📂 Found existing FAISS index at: {FAISS_INDEX_PATH}")
            use_existing = input("Use existing index? (y/n): ").strip().lower()
            
            if use_existing in ['y', 'yes']:
                rag_app.load_existing_vectorstore()
                rag_app.setup_qa_chain()
                rag_app.interactive_query_loop()
                return
        
        # Get PDF files to process
        print("\n📄 Document Processing")
        print("Enter PDF file paths (one per line, empty line to finish):")
        
        pdf_paths = []
        while True:
            path = input("PDF path: ").strip()
            if not path:
                break
            pdf_paths.append(path)
        
        if not pdf_paths:
            print("❌ No PDF paths provided. Exiting...")
            return
        
        # Process documents
        docs = rag_app.load_documents(pdf_paths)
        rag_app.create_vectorstore(docs)
        rag_app.setup_qa_chain()
        rag_app.interactive_query_loop()
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
