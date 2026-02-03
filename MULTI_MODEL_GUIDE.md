# Multi-Model Query Feature Guide

## Overview

The RAG Assistant now supports **Multi-Model Queries**, allowing you to get responses from multiple AI models simultaneously for comprehensive insights on your questions.

## How It Works

1. **Parallel Processing**: When you enable multi-model mode, your question is sent to multiple specialized AI models at the same time
2. **Specialized Models**: Each model brings its unique expertise:
   - **Technical**: Technical and scientific questions (llama3.1)
   - **Creative**: Creative writing and storytelling (llama3.2) 
   - **Code**: Programming and code-related questions (codellama)
   - **Analysis**: Data analysis and reasoning (llama3.1)
   - **Chat**: Conversational and general questions (llama3.2:3b)
   - **Reasoning**: Complex reasoning and logic (qwen2.5)
   - **Default**: General purpose AI assistant (llama3.2)

3. **Combined Response**: All model responses are intelligently combined into a comprehensive answer

## Using Multi-Model Queries

### Mobile Interface

1. **Enable Multi-Model Mode**:
   - Look for the "Multi-Model Query" toggle switch above the input area
   - Tap the switch to turn it on (it will turn blue)

2. **Select Models**:
   - Once enabled, you'll see model selection buttons
   - Tap on the models you want to include (selected models turn blue)
   - You can select multiple models for different perspectives

3. **Ask Your Question**:
   - Type your question normally
   - The question will be processed by all selected models in parallel

4. **View Results**:
   - You'll get a combined response showing perspectives from each selected model
   - Each model's contribution is clearly labeled

### Desktop Interface

1. **Enable Multi-Model Mode**:
   - Find the "Multi-Model Query" toggle in the input area
   - Click the switch to enable it

2. **Select Models**:
   - Choose from the available model cards
   - Selected models will be highlighted
   - Each card shows the model type and underlying AI model

3. **Submit Query**:
   - Type your question and press Enter or click Send
   - All selected models will process your question simultaneously

## Example Use Cases

### 1. Comprehensive Car Analysis
**Question**: "Explain how electric cars work"
**Selected Models**: Technical + Creative
**Result**: Technical explanation + Creative storytelling about the future of electric vehicles

### 2. Code Problem Solving
**Question**: "How to optimize a Python sorting algorithm"
**Selected Models**: Code + Analysis
**Result**: Technical implementation + Performance analysis and optimization strategies

### 3. Business Decision Making
**Question**: "Should we implement remote work policies?"
**Selected Models**: Analysis + Chat
**Result**: Data-driven analysis + Conversational pros/cons discussion

### 4. Creative Writing with Technical Accuracy
**Question**: "Write a story about space exploration"
**Selected Models**: Creative + Technical
**Result**: Engaging narrative + Scientifically accurate space details

## Benefits

- **Multiple Perspectives**: Get different viewpoints on complex topics
- **Specialized Expertise**: Each model brings domain-specific knowledge
- **Time Efficient**: All models process your question simultaneously
- **Comprehensive Answers**: Rich, multi-faceted responses
- **Validation**: Cross-reference information from multiple AI systems

## API Endpoint

For developers, the multi-model feature is available via:

```
POST /api/ask-multi
Content-Type: application/json

{
    "question": "Your question here",
    "model_types": ["technical", "creative", "code"]
}
```

**Response includes**:
- Combined formatted answer
- Individual model results
- Sources from all models
- Performance metrics

## Tips for Best Results

1. **Choose Relevant Models**: Select models that match your question's domain
2. **Use 2-4 Models**: Too many models can make responses overwhelming
3. **Specific Questions**: More specific questions yield better multi-model insights
4. **Mixed Domains**: Great for questions that span multiple areas of expertise

## Technical Details

- **Parallel Processing**: Uses ThreadPoolExecutor for concurrent model queries
- **Thread Safety**: Each model query runs in its own thread
- **Timeout Protection**: 30-second timeout per model to prevent hanging
- **Error Handling**: Failed models are reported without breaking the entire query
- **Performance**: Typically 2-3x faster than sequential model queries

The multi-model feature transforms your RAG assistant into a collaborative AI team, providing richer and more comprehensive responses to your questions!
