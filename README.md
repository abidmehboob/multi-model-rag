# RAG Application with LLaMA using Ollama

This project implements a Retrieval-Augmented Generation (RAG) system using LLaMA models through Ollama, based on the tutorial from [Medium](https://medium.com/@danushidk507/rag-with-llama-using-ollama-a-deep-dive-into-retrieval-augmented-generation-c58b9a1cfcd3).

## Features

- 📄 PDF document processing and chunking
- 🔍 FAISS vector storage for efficient similarity search
- 🤖 LLaMA model integration via Ollama
- 💬 Interactive question-answering interface
- 🔧 Configurable parameters
- 📚 Multi-document support
- 💾 Persistent vector storage

## Prerequisites

### System Requirements
- Python 3.8+
- Windows/Linux/macOS
- At least 8GB RAM (16GB recommended)
- GPU (optional, for faster processing)

### Dependencies

Install Python packages:
```bash
py -m pip install -r requirements.txt
```

### Ollama Setup

1. **Install Ollama:**
   - Windows/macOS: Download from [https://ollama.com/download](https://ollama.com/download)
   - Linux: Run `curl -fsSL https://ollama.com/install.sh | sh`

2. **Start Ollama service:**
   ```bash
   ollama serve
   ```

3. **Pull LLaMA model:**
   ```bash
   ollama pull llama3.1
   ```

   Available models:
   - `llama3.1` (recommended, 8B parameters)
   - `llama3.1:70b` (larger, better quality)
   - `llama2` (alternative)

## Quick Start

### Option 1: Automated Setup
```bash
python setup_ollama.py
```

### Option 2: Manual Setup

1. **Install dependencies:**
   ```bash
   py -m pip install -r requirements.txt
   ```

2. **Set up Ollama and models** (see Prerequisites)

3. **Run the basic application:**
   ```bash
   python rag_app.py
   ```

4. **Or run the enhanced version:**
   ```bash
   python enhanced_rag_app.py
   ```

## Usage

### Basic RAG Application (`rag_app.py`)

```python
from rag_app import RAGApplication

# Initialize
rag_app = RAGApplication()

# Process a document
rag_app.load_and_process_document("path/to/your/document.pdf")

# Setup QA chain
rag_app.setup_qa_chain()

# Ask questions
answer = rag_app.query("What is this document about?")

# Interactive mode
rag_app.interactive_query_loop()
```

### Enhanced RAG Application (`enhanced_rag_app.py`)

Features:
- ✨ Better error handling
- 📊 Progress indicators
- 🔧 Configuration support
- 📚 Multi-document processing
- 📍 Source document tracking

```python
from enhanced_rag_app import EnhancedRAGApplication

# Initialize
rag_app = EnhancedRAGApplication()

# Process multiple documents
docs = rag_app.load_documents([
    "document1.pdf",
    "document2.pdf"
])

# Create vector store
rag_app.create_vectorstore(docs)

# Start interactive session
rag_app.interactive_query_loop()
```

## Configuration

Modify `config.py` to customize:

```python
# Model Configuration
OLLAMA_MODEL = "llama3.1"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Text Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 30

# Hardware
DEVICE = "cpu"  # or "cuda" for GPU

# Retrieval
SEARCH_KWARGS = {"k": 4}  # Number of documents to retrieve
```

## Project Structure

```
rag/
├── requirements.txt          # Python dependencies
├── config.py                # Configuration settings
├── rag_app.py               # Basic RAG application
├── enhanced_rag_app.py      # Enhanced RAG application
├── setup_ollama.py          # Ollama setup script
├── faiss_index/             # Vector store (created after first run)
└── README.md                # This file
```

## How It Works

### 1. Document Processing
- Load PDF documents using PyPDFLoader
- Split into chunks with overlap for better context
- Generate embeddings using sentence-transformers

### 2. Vector Storage
- Store embeddings in FAISS for efficient similarity search
- Save index persistently for reuse

### 3. Retrieval
- Find most relevant document chunks for queries
- Use semantic similarity matching

### 4. Generation
- Feed retrieved context to LLaMA model
- Generate accurate, grounded responses

## Example Usage

```bash
$ python enhanced_rag_app.py

🦙 RAG with LLaMA using Ollama - Enhanced Version
============================================================
🚀 Initializing Enhanced RAG Application...
📊 Setting up embedding model: sentence-transformers/all-mpnet-base-v2
✅ Embedding model loaded successfully
🤖 Setting up LLaMA model: llama3.1
✅ LLaMA model loaded successfully

📄 Document Processing
Enter PDF file paths (one per line, empty line to finish):
PDF path: sample_document.pdf
PDF path: 

📄 Loading document: sample_document.pdf
📖 Loaded 10 pages from sample_document.pdf
✂️ Splitting 10 pages into chunks...
📚 Created 45 document chunks
🔍 Creating vector embeddings and FAISS index...
💾 Saving vector store to 'faiss_index'...
✅ Vector store created and retriever configured
🔗 Setting up RetrievalQA chain...
✅ QA chain setup complete

======================================================================
🎯 RAG Application Ready!
💬 Type your questions below
📝 Commands: 'exit', 'quit', 'q' to quit
======================================================================

🤔 Your question: What is the main topic of this document?

❓ Question: What is the main topic of this document?
🔍 Searching for relevant information...
💡 Answer: Based on the document content, the main topic appears to be...

📚 Sources (3 documents):
  1. sample_document.pdf (Page: 1)
  2. sample_document.pdf (Page: 2)
  3. sample_document.pdf (Page: 5)
```

## Troubleshooting

### Common Issues

1. **"ERROR: Could not find a version that satisfies the requirement faiss-gpu"**
   - Solution: Use `faiss-cpu` instead (already in requirements.txt)

2. **"Connection refused" when connecting to Ollama**
   - Solution: Start Ollama service: `ollama serve`

3. **Model not found**
   - Solution: Pull the model: `ollama pull llama3.1`

4. **Out of memory errors**
   - Solution: Use smaller chunk sizes or switch to CPU processing

### Performance Tips

1. **GPU Acceleration:**
   ```python
   # In config.py
   DEVICE = "cuda"  # Requires CUDA-compatible GPU
   ```

2. **Faster Embedding Model:**
   ```python
   EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```

3. **Optimized Chunk Size:**
   ```python
   CHUNK_SIZE = 500  # Smaller chunks for faster processing
   CHUNK_OVERLAP = 50  # More overlap for better context
   ```

## Advanced Features

### Custom Prompts
Modify the LLaMA prompt by customizing the RetrievalQA chain:

```python
from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
Answer: """

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)
```

### Multiple Models
Switch between different Ollama models:

```python
# List available models
ollama list

# Use different model
rag_app = RAGApplication(model_name="llama2")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is based on the tutorial by DhanushKumar and is provided as-is for educational purposes.

## References

- [Original Medium Article](https://medium.com/@danushidk507/rag-with-llama-using-ollama-a-deep-dive-into-retrieval-augmented-generation-c58b9a1cfcd3)
- [Ollama Documentation](https://ollama.com)
- [LangChain Documentation](https://langchain.readthedocs.io)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net)
