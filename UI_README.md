# Multi-Model RAG Application - Web & Mobile UI

## 🚀 Features

### 📤 **Document Up### 🔗 API Endpoints

### Upload Documents
```
P## 🎮 Usage Examples

### Document Upload
1. **Web Interface**: Drag files to the upload area in the sidebar
2. **Mobile Interface**: Tap the upload section in the side menu
3. **Supported Files**: PDF documents, text files (.txt)
4. **File Size**: Maximum 16MB per file
5. **Processing**: Files are automatically chunked and added to the knowledge base

### Question Types /api/upload
Content-Type: multipart/form-data

Form data:
- files: Multiple files (PDF, TXT)
```
Returns upload status and processing results.

### Get Documents
```
GET /api/documents
```
Returns information about loaded documents and sources.

### Health Checkd & Processing**
- **Drag & Drop**: Easy file upload with drag-and-drop interface
- **Multiple Formats**: Support for PDF and TXT files
- **Real-time Processing**: Instant chunking and vector store integration
- **File Validation**: Automatic file type and size validation (16MB max)
- **Progress Feedback**: Upload status and processing results

### 🎯 Smart Model Selection
- **Technical questions** → llama3.1 (low temperature for precision)
- **Code questions** → codellama (zero temperature for accuracy)
- **Creative questions** → gemma2 (high temperature for creativity)
- **Analysis questions** → llama3.1 (medium temperature for reasoning)
- **General questions** → gemma2 (balanced temperature)

### 🔗 SSH-Based Dynamic Model Management
- Automatically checks available models on Ollama server
- Pulls missing models via SSH when needed
- Real-time model availability monitoring

### 📱 Multiple Interfaces
1. **Web UI** - Full desktop experience with sidebar statistics
2. **Mobile UI** - Touch-optimized responsive design
3. **Command Line** - Traditional CLI interface

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install flask flask-cors scikit-learn numpy paramiko
```

### Optional for PDF support:
```bash
pip install langchain langchain-community
```

## 🎮 Running the Application

### 1. Web & Mobile Interface
```bash
cd "d:\Allianz_Onboarding_GenAI specialist\rag"
py app.py
```

**Access URLs:**
- 🖥️ **Web UI**: http://localhost:5000/
- 📱 **Mobile UI**: http://localhost:5000/mobile
- 🔗 **API**: http://localhost:5000/api/

### 2. Command Line Interface
```bash
py working_rag_app.py
```

## 📱 Mobile UI Features

### Touch-Optimized Design
- Swipe-friendly interface
- Tap-to-send messages
- Auto-resizing input field
- Side menu with statistics

### Mobile-Specific Features
- **Status Indicator**: Shows connection health
- **Typing Indicators**: Visual feedback during processing
- **Model Badges**: Shows which AI model was used
- **Source Display**: Compact source information
- **Statistics Panel**: Question count, sources used
- **Chat History**: Persistent conversation history

## 🖥️ Web UI Features

### Desktop Interface
- **Split Layout**: Sidebar with models & stats + main chat area
- **Real-time Stats**: Live question count, document usage
- **Model Information**: Active model configurations display
- **Advanced Chat**: Rich formatting, source previews
- **Clear History**: One-click conversation reset

### Key Components
- **Smart Input**: Auto-expanding textarea
- **Model Detection**: Visual indicators for model selection
- **Source Panel**: Detailed document source information
- **Status Dashboard**: System health monitoring

## 🔗 API Endpoints

### Health Check
```
GET /api/health
```
Returns system status and RAG initialization state.

### Get Available Models
```
GET /api/models
```
Returns configured models and their availability.

### Ask Question
```
POST /api/ask
Content-Type: application/json

{
  "question": "What is machine learning?"
}
```

### Get Conversation History
```
GET /api/history?limit=20
```

### Clear History
```
POST /api/clear-history
```

## 🎯 Usage Examples

### Technical Questions
**Question**: "Explain neural network backpropagation"
- **Model Used**: llama3.1 (technical)
- **Temperature**: 0.1 (precise)

### Code Questions
**Question**: "Create a Python function for binary search"
- **Model Used**: codellama (code)
- **Temperature**: 0.0 (exact)

### Creative Questions
**Question**: "Write a story about AI helping humans"
- **Model Used**: gemma2 (creative)
- **Temperature**: 0.7 (imaginative)

## 📊 Model Configuration

```python
MULTI_MODEL_CONFIG = {
    "technical": {
        "model": "llama3.1",
        "temperature": 0.1,
        "description": "Technical and scientific questions"
    },
    "code": {
        "model": "codellama",
        "temperature": 0.0,
        "description": "Code generation and programming"
    },
    "creative": {
        "model": "gemma2",
        "temperature": 0.7,
        "description": "Creative writing and storytelling"
    },
    "analysis": {
        "model": "llama3.1",
        "temperature": 0.3,
        "description": "Data analysis and reasoning"
    },
    "default": {
        "model": "gemma2",
        "temperature": 0.2,
        "description": "General purpose AI assistant"
    }
}
```

## 🔧 SSH Configuration

The application automatically connects to your Ollama server via SSH to:
- Check available models
- Pull missing models automatically
- Monitor server status

**SSH Settings** (configured in `working_rag_app.py`):
```python
SSH_CONFIG = {
    "host": "20.185.83.16",
    "port": 22,
    "user": "llama",
    "password": "Capadmin@024"
}
```

## 🎨 UI Screenshots

### Web Interface
- Clean, modern design with gradient backgrounds
- Responsive layout that works on desktop and tablets
- Real-time model information display
- Interactive statistics dashboard

### Mobile Interface  
- Native mobile app feel
- Touch-optimized controls
- Slide-out menu for settings
- Compact message bubbles
- Typing indicators and status lights

## 🚨 Troubleshooting

### Connection Issues
1. Check if Ollama server is running on `20.185.83.16:8080`
2. Verify SSH credentials for model management
3. Ensure required models are available

### Model Loading Issues
1. Check SSH connectivity to Ollama server
2. Verify disk space for model downloads
3. Check internet connection for model pulling

### UI Issues
1. Clear browser cache and reload
2. Check if Flask server is running
3. Verify all dependencies are installed

## 🔄 Development Mode

For development with auto-reload:
```bash
set FLASK_ENV=development
py app.py
```

## 📝 Notes

- The application creates sample AI/ML documents if no PDFs are provided
- Chat history is stored in memory (clears on restart)
- Vector store is saved to disk for persistence
- Mobile UI is PWA-ready for app-like experience

## 🌟 Features Coming Soon

- [ ] File upload via web interface
- [ ] Voice input for mobile
- [ ] Dark/Light theme toggle
- [ ] Export chat history
- [ ] Multi-language support
- [ ] Real-time collaboration
