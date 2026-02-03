"""
RAG with LLaMA using Ollama - Implementation
Based on: https://medium.com/@danushidk507/rag-with-llama-using-ollama-a-deep-dive-into-retrieval-augmented-generation-c58b9a1cfcd3
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

class RAGApplication:
    def __init__(self, pdf_path=None, model_name="llama3.1"):
        """
        Initialize the RAG application
        
        Args:
            pdf_path (str): Path to the PDF document to process
            model_name (str): Name of the Ollama model to use
        """
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        
        # Initialize embedding model
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model_kwargs = {"device": "cpu"}  # Change to "cuda" if GPU available
        
        print("Initializing RAG Application...")
        self._setup_embeddings()
        self._setup_llm()
    
    def _setup_embeddings(self):
        """Setup the embedding model"""
        print("Setting up embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=self.model_kwargs
        )
        print(f"Embedding model loaded: {self.embedding_model_name}")
    
    def _setup_llm(self):
        """Setup the LLaMA model using Ollama"""
        print("Setting up LLaMA model...")
        try:
            self.llm = Ollama(model=self.model_name)
            # Test the model
            test_response = self.llm.invoke("Hello, are you working?")
            print(f"LLaMA model loaded successfully: {self.model_name}")
            print(f"Test response: {test_response[:100]}...")
        except Exception as e:
            print(f"Error setting up LLaMA model: {e}")
            print("Make sure Ollama is installed and the model is available.")
            print("Install Ollama: https://ollama.com/download")
            print(f"Then run: ollama pull {self.model_name}")
    
    def load_and_process_document(self, pdf_path=None):
        """
        Load and process a PDF document
        
        Args:
            pdf_path (str): Path to the PDF document
        """
        if pdf_path:
            self.pdf_path = pdf_path
        
        if not self.pdf_path:
            raise ValueError("No PDF path provided")
        
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        print(f"Loading document: {self.pdf_path}")
        
        # 1. Data Ingestion - Load the PDF document
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from the document")
        
        # Split the document into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=30,
            separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)
        print(f"Split into {len(docs)} chunks")
        
        # 2. Data Embedding and Storage with FAISS
        print("Creating vector embeddings and FAISS index...")
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        
        # Save the vector store
        print("Saving vector store...")
        self.vectorstore.save_local("faiss_index")
        print("Vector store saved to 'faiss_index' directory")
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever()
        print("Retriever created successfully")
        
        return docs
    
    def load_existing_vectorstore(self, index_path="faiss_index"):
        """
        Load an existing FAISS vector store
        
        Args:
            index_path (str): Path to the FAISS index directory
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        print(f"Loading existing vector store from: {index_path}")
        self.vectorstore = FAISS.load_local(
            index_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()
        print("Vector store loaded successfully")
    
    def setup_qa_chain(self):
        """Setup the RetrievalQA chain"""
        if not self.retriever:
            raise ValueError("Retriever not initialized. Load a document first.")
        
        if not self.llm:
            raise ValueError("LLM not initialized.")
        
        print("Setting up RetrievalQA chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )
        print("QA chain setup complete")
    
    def query(self, question):
        """
        Ask a question and get an answer based on the documents
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The answer from the RAG system
        """
        if not self.qa_chain:
            raise ValueError("QA chain not setup. Call setup_qa_chain() first.")
        
        print(f"\nQuestion: {question}")
        print("Searching for relevant information...")
        
        result = self.qa_chain.run(question)
        print(f"Answer: {result}")
        
        return result
    
    def interactive_query_loop(self):
        """Start an interactive query loop"""
        if not self.qa_chain:
            print("Setting up QA chain...")
            self.setup_qa_chain()
        
        print("\n" + "="*60)
        print("RAG Application Ready!")
        print("Type your questions below (type 'exit' to quit)")
        print("="*60)
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                result = self.query(query)
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function to run the RAG application"""
    print("RAG with LLaMA using Ollama")
    print("=" * 40)
    
    # Initialize the application
    rag_app = RAGApplication()
    
    # Check if there's an existing vector store
    if os.path.exists("faiss_index"):
        print("Found existing FAISS index. Loading...")
        rag_app.load_existing_vectorstore()
        rag_app.setup_qa_chain()
        rag_app.interactive_query_loop()
    else:
        # Ask for PDF path
        pdf_path = input("Enter the path to your PDF document: ").strip()
        
        if not pdf_path:
            print("No PDF path provided. Exiting...")
            return
        
        try:
            # Process the document
            docs = rag_app.load_and_process_document(pdf_path)
            
            # Setup QA chain
            rag_app.setup_qa_chain()
            
            # Start interactive loop
            rag_app.interactive_query_loop()
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
