"""
Configuration file for RAG Application
"""

# Model Configuration
OLLAMA_MODEL = "gemma2:latest"  # Updated to match server model name
OLLAMA_API_URL = "http://20.185.83.16:8080/"
OLLAMA_API_KEY = "aie93JaTv1GW1AP4IIUSqeecV22HgpcQ6WlgWNyfx2HflkY5hTw19JDbT90ViKcZaZ6lpjOo3YIGgpkG7Zb8jEKvdM5Ymnq9jPm79osLppCebwJ7WdWTwWq3Rf15NDxm"
DEFAULT_TEMPERATURE = 0.2

# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Alternative embedding models you can try:
# "sentence-transformers/all-MiniLM-L6-v2"  # Faster, smaller
# "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # Good for Q&A

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 30
SEPARATOR = "\n"

# Device Configuration
DEVICE = "cpu"  # Change to "cuda" if you have GPU support

# FAISS Index Configuration
FAISS_INDEX_PATH = "faiss_index"

# Retrieval Configuration
SEARCH_TYPE = "similarity"
SEARCH_KWARGS = {"k": 4}  # Number of documents to retrieve

# Chain Configuration
CHAIN_TYPE = "stuff"  # Can be "stuff", "map_reduce", "refine", or "map_rerank"
