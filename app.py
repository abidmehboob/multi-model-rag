"""
Flask Web API for Multi-Model RAG Application
Provides REST endpoints for questions and answers
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import logging
from datetime import datetime
from working_rag_app import SimpleRAG, SimpleDocument, SimpleTFIDFVectorStore, MULTI_MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global RAG instance
rag_instance = None
conversation_history = []

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_rag():
    """Initialize RAG system with improved error handling"""
    global rag_instance
    try:
        logger.info("🚀 Initializing RAG system...")
        
        # Create RAG instance with timeout and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rag_instance = SimpleRAG()
                break
            except Exception as e:
                logger.warning(f"⚠️ RAG initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("🔄 Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                else:
                    raise Exception(f"Failed to initialize RAG after {max_retries} attempts")
        
        # Load existing vector store or create sample documents
        vectorstore_path = "simple_vectorstore.pkl"
        if os.path.exists(vectorstore_path):
            logger.info("📂 Loading existing vector store...")
            try:
                rag_instance.load_vectorstore()
                logger.info("✅ Vector store loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load vector store: {e}")
                logger.info("🔄 Creating new sample documents...")
                docs = rag_instance.create_sample_documents()
                rag_instance.process_documents(docs)
                rag_instance.save_vectorstore()
        else:
            logger.info("📚 Creating sample documents...")
            docs = rag_instance.create_sample_documents()
            rag_instance.process_documents(docs)
            rag_instance.save_vectorstore()
        
        # Validate that at least one model is working
        if not rag_instance.llm_models:
            logger.error("❌ No working models found!")
            return False
            
        logger.info(f"✅ RAG system initialized with {len(rag_instance.llm_models)} working models")
        
        # Log available models
        working_models = []
        for prompt_type, model in rag_instance.llm_models.items():
            working_models.append(f"{prompt_type}")
        
        logger.info(f"🤖 Available prompt types: {', '.join(working_models)}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        return False

@app.route('/')
def index():
    """Serve main web interface"""
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    """Serve mobile interface"""
    return render_template('mobile.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed server status"""
    server_status = {
        'flask_app': 'healthy',
        'rag_initialized': rag_instance is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    if rag_instance:
        server_status.update({
            'models_loaded': len(rag_instance.llm_models),
            'available_models': list(rag_instance.llm_models.keys()),
            'vectorstore_loaded': rag_instance.vectorstore is not None
        })
        
        # Test Ollama server connectivity
        try:
            # Quick test to see if we can reach the Ollama server
            from working_rag_app import OLLAMA_API_URL
            import requests
            response = requests.get(f"{OLLAMA_API_URL}api/tags", timeout=5)
            server_status['ollama_server'] = 'reachable' if response.status_code == 200 else 'unreachable'
            server_status['ollama_url'] = OLLAMA_API_URL
        except Exception as e:
            server_status['ollama_server'] = 'unreachable'
            server_status['ollama_error'] = str(e)
            server_status['ollama_url'] = OLLAMA_API_URL
    
    return jsonify(server_status)

@app.route('/api/server-config', methods=['GET'])
def get_server_config():
    """Get current server configuration"""
    from working_rag_app import OLLAMA_API_URL, SSH_CONFIG
    
    return jsonify({
        'ollama_url': OLLAMA_API_URL,
        'ssh_host': SSH_CONFIG.get('host', 'Unknown'),
        'is_remote': '20.185.83.16' in OLLAMA_API_URL,
        'connection_status': 'unknown'  # Will be updated by health check
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available model configurations"""
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    from working_rag_app import MULTI_MODEL_CONFIG
    
    # Combine built-in and custom models
    all_models = MULTI_MODEL_CONFIG.copy()
    all_models.update(rag_instance.custom_prompt_types)
    
    return jsonify({
        'models': all_models,
        'available_models': list(rag_instance.llm_models.keys()),
        'builtin_types': list(MULTI_MODEL_CONFIG.keys()),
        'custom_types': list(rag_instance.custom_prompt_types.keys())
    })

@app.route('/api/prompt-types', methods=['GET'])
def get_prompt_types():
    """Get all available prompt types"""
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    from working_rag_app import MULTI_MODEL_CONFIG
    
    builtin_types = []
    for ptype, config in MULTI_MODEL_CONFIG.items():
        builtin_types.append({
            'name': ptype,
            'type': 'builtin',
            'model': config['model'],
            'temperature': config['temperature'],
            'description': config['description'],
            'available': ptype in rag_instance.llm_models
        })
    
    custom_types = []
    for ptype, config in rag_instance.custom_prompt_types.items():
        custom_types.append({
            'name': ptype,
            'type': 'custom',
            'model': config['model'],
            'temperature': config['temperature'],
            'description': config['description'],
            'keywords': config['keywords'],
            'available': ptype in rag_instance.llm_models
        })
    
    return jsonify({
        'builtin_types': builtin_types,
        'custom_types': custom_types,
        'total': len(builtin_types) + len(custom_types)
    })

@app.route('/api/prompt-types', methods=['POST'])
def add_prompt_type():
    """Add a new custom prompt type with automatic model pulling"""
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['name', 'model', 'description', 'keywords']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400
        
        type_name = data['name'].strip().lower()
        model_name = data['model'].strip()
        temperature = float(data.get('temperature', 0.2))
        description = data['description'].strip()
        keywords = [kw.strip().lower() for kw in data['keywords'] if kw.strip()]
        
        # Validate values
        if temperature < 0 or temperature > 1:
            return jsonify({'error': 'Temperature must be between 0 and 1'}), 400
        
        if not keywords:
            return jsonify({'error': 'At least one keyword is required'}), 400
        
        # Check for existing types
        if type_name in rag_instance.custom_prompt_types:
            return jsonify({'error': f'Custom type "{type_name}" already exists'}), 409
        
        # Check if type name conflicts with built-in types
        from working_rag_app import MULTI_MODEL_CONFIG
        if type_name in MULTI_MODEL_CONFIG:
            return jsonify({'error': f'Type name "{type_name}" conflicts with built-in type'}), 409
        
        logger.info(f"Adding custom prompt type: {type_name} with model: {model_name}")
        
        # Store original available models to detect if we pulled a new one
        original_models = set(rag_instance.model_manager.available_models)
        
        # Add the custom type (this will auto-pull the model if needed)
        success = rag_instance.add_custom_prompt_type(
            type_name, model_name, temperature, description, keywords
        )
        
        if success:
            # Check if a new model was pulled
            new_models = set(rag_instance.model_manager.available_models)
            model_was_pulled = len(new_models) > len(original_models)
            
            response_data = {
                'message': f'Successfully added custom prompt type "{type_name}"',
                'type_name': type_name,
                'model_name': model_name,
                'temperature': temperature,
                'description': description,
                'keywords': keywords,
                'model_pulled': model_was_pulled,
                'status': 'Model pulled and initialized' if model_was_pulled else 'Model was already available'
            }
            
            if model_was_pulled:
                response_data['info'] = f'Model "{model_name}" was automatically downloaded from Ollama repository'
            
            return jsonify(response_data), 201
        else:
            return jsonify({
                'error': 'Failed to add custom prompt type',
                'details': 'Model may not exist in Ollama repository or failed to initialize'
            }), 500
            
    except ValueError as e:
        return jsonify({'error': f'Invalid temperature value: {e}'}), 400
    except Exception as e:
        logger.error(f"Error adding custom prompt type: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/prompt-types/<type_name>', methods=['DELETE'])
def remove_prompt_type(type_name):
    """Remove a custom prompt type"""
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        success = rag_instance.remove_custom_prompt_type(type_name)
        
        if success:
            return jsonify({'message': f'Successfully removed custom prompt type "{type_name}"'})
        else:
            return jsonify({'error': f'Custom prompt type "{type_name}" not found'}), 404
            
    except Exception as e:
        logger.error(f"Error removing custom prompt type: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/prompt-types/<type_name>', methods=['PUT'])
def modify_prompt_type(type_name):
    """Modify an existing custom prompt type"""
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        # Check if type exists
        if type_name not in rag_instance.custom_prompt_types:
            return jsonify({'error': f'Custom type "{type_name}" not found'}), 404
        
        data = request.json
        current_config = rag_instance.custom_prompt_types[type_name]
        
        # Get new values or keep current ones
        new_model = data.get('model', current_config['model']).strip()
        new_temperature = float(data.get('temperature', current_config['temperature']))
        new_description = data.get('description', current_config['description']).strip()
        new_keywords = data.get('keywords', current_config['keywords'])
        
        # Validate new values
        if new_temperature < 0 or new_temperature > 1:
            return jsonify({'error': 'Temperature must be between 0 and 1'}), 400
        
        if isinstance(new_keywords, str):
            new_keywords = [kw.strip().lower() for kw in new_keywords.split(',') if kw.strip()]
        
        if not new_keywords:
            return jsonify({'error': 'At least one keyword is required'}), 400
        
        logger.info(f"Modifying custom prompt type: {type_name}")
        
        # Remove old model instance
        if type_name in rag_instance.llm_models:
            del rag_instance.llm_models[type_name]
        
        # Update the custom type
        success = rag_instance.add_custom_prompt_type(
            type_name, new_model, new_temperature, new_description, new_keywords
        )
        
        if success:
            return jsonify({
                'message': f'Successfully modified custom prompt type "{type_name}"',
                'type_name': type_name,
                'model_name': new_model,
                'temperature': new_temperature,
                'description': new_description,
                'keywords': new_keywords
            })
        else:
            return jsonify({'error': 'Failed to modify custom prompt type'}), 500
            
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error modifying prompt type: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/map-model', methods=['POST'])
def map_model_to_type():
    """Map an available Ollama model to a new custom type"""
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['model', 'type_name']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400
        
        model_name = data['model'].strip()
        type_name = data['type_name'].strip().lower()
        
        # Check if type already exists
        if type_name in rag_instance.custom_prompt_types or type_name in MULTI_MODEL_CONFIG:
            return jsonify({'error': f'Type "{type_name}" already exists'}), 409
        
        # Refresh available models
        rag_instance.model_manager.refresh_available_models()
        
        # Check if model is available
        base_model = model_name.split(':')[0]
        if base_model not in rag_instance.model_manager.available_models:
            return jsonify({'error': f'Model "{model_name}" not found on server'}), 404
        
        # Set default values based on model type
        default_temp = 0.2
        if 'code' in model_name.lower():
            default_temp = 0.0
        elif any(x in model_name.lower() for x in ['creative', 'story']):
            default_temp = 0.7
        
        temperature = float(data.get('temperature', default_temp))
        
        # Generate smart defaults
        description = data.get('description', f"Custom type using {model_name} model")
        
        # Parse keywords
        keywords = data.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [kw.strip().lower() for kw in keywords.split(',') if kw.strip()]
        
        # Set default keywords if none provided
        if not keywords:
            if 'code' in model_name.lower():
                keywords = ['code', 'programming', 'development']
            elif 'qwen' in model_name.lower():
                keywords = ['analysis', 'reasoning', 'logic']
            else:
                keywords = [type_name, base_model]
        
        # Validate values
        if temperature < 0 or temperature > 1:
            return jsonify({'error': 'Temperature must be between 0 and 1'}), 400
        
        if not keywords:
            return jsonify({'error': 'At least one keyword is required'}), 400
        
        logger.info(f"Mapping model {model_name} to type {type_name}")
        
        # Create the mapping
        success = rag_instance.add_custom_prompt_type(
            type_name, model_name, temperature, description, keywords
        )
        
        if success:
            return jsonify({
                'message': f'Successfully mapped model "{model_name}" to type "{type_name}"',
                'type_name': type_name,
                'model_name': model_name,
                'temperature': temperature,
                'description': description,
                'keywords': keywords
            }), 201
        else:
            return jsonify({'error': 'Failed to create mapping'}), 500
            
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error mapping model: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """Get list of available Ollama models with mapping status"""
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        # Refresh model list
        rag_instance.model_manager.refresh_available_models()
        
        models_info = []
        for model in sorted(rag_instance.model_manager.available_models):
            # Check if model is mapped to any types
            mapped_types = []
            
            # Check custom types
            for type_name, config in rag_instance.custom_prompt_types.items():
                if config['model'].split(':')[0] == model:
                    mapped_types.append({
                        'name': type_name,
                        'type': 'custom',
                        'temperature': config['temperature'],
                        'description': config['description']
                    })
            
            # Check built-in types
            for type_name, config in MULTI_MODEL_CONFIG.items():
                if config['model'].split(':')[0] == model:
                    mapped_types.append({
                        'name': type_name,
                        'type': 'built-in',
                        'temperature': config['temperature'],
                        'description': config['description']
                    })
            
            models_info.append({
                'name': model,
                'is_mapped': len(mapped_types) > 0,
                'mapped_types': mapped_types,
                'suggested_temperature': 0.0 if 'code' in model.lower() else 0.7 if any(x in model.lower() for x in ['creative', 'story']) else 0.2
            })
        
        return jsonify({
            'models': models_info,
            'total_models': len(models_info),
            'mapped_count': len([m for m in models_info if m['is_mapped']]),
            'unmapped_count': len([m for m in models_info if not m['is_mapped']])
        })
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Process question and return answer with automatic multi-model detection"""
    global conversation_history
    
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Check if question contains model-specific sections
        parsed_sections = parse_model_specific_question(question)
        
        if parsed_sections:
            # Question contains model-specific sections, process as parsed multi-model
            logger.info(f"Auto-detected model-specific sections: {[s['model_type'] for s in parsed_sections]}")
            
            # Process each section with its specific model
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def query_specific_section(section):
                model_type = section['model_type']
                section_question = section['question']
                
                # Try with retry logic for connection issues
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        result = rag_instance.query_documents(section_question, force_prompt_type=model_type)
                        return {
                            'model_type': model_type,
                            'section_question': section_question,
                            'success': True,
                            'answer': result['answer'],
                            'sources': result.get('sources', []),
                            'model_config': result.get('model_config', {}),
                            'prompt_type': result.get('prompt_type', model_type)
                        }
                    except Exception as e:
                        error_msg = str(e)
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed for {model_type}: {error_msg}. Retrying...")
                            import time
                            time.sleep(2)  # Wait 2 seconds before retry
                            continue
                        
                        # Final attempt failed, provide detailed error info
                        if "timeout" in error_msg.lower():
                            fallback_answer = f"⏰ **Connection Timeout**: The {model_type.title()} model server is experiencing delays. This usually means the server is overloaded or the network connection is slow.\n\n**Suggested Actions:**\n• Try again in a few moments\n• Check if the Ollama server is running\n• Consider using a local Ollama installation"
                        elif "404" in error_msg or "not found" in error_msg.lower():
                            fallback_answer = f"🔍 **Model Not Available**: The {model_type.title()} model endpoint was not found on the server.\n\n**Possible Causes:**\n• The model may not be installed on the remote server\n• The API endpoint configuration may be incorrect\n• The server may be temporarily unavailable\n\n**Suggested Actions:**\n• Verify the model is available: `ollama list`\n• Try using a different model type\n• Contact your system administrator"
                        elif "connection" in error_msg.lower() or "pool" in error_msg.lower():
                            fallback_answer = f"🌐 **Connection Error**: Unable to connect to the {model_type.title()} model server.\n\n**Network Issue Details:**\n• Server: 20.185.83.16:8080\n• Error: {error_msg}\n\n**Suggested Actions:**\n• Check your internet connection\n• Verify VPN/proxy settings\n• Try switching to localhost if running Ollama locally"
                        else:
                            fallback_answer = f"❌ **{model_type.title()} Model Error**: {error_msg}\n\n**Suggested Actions:**\n• Try refreshing the page\n• Use a different model type\n• Contact support if the issue persists"
                        
                        return {
                            'model_type': model_type,
                            'section_question': section_question,
                            'success': False,
                            'error': error_msg,
                            'answer': fallback_answer,
                            'retry_attempts': max_retries
                        }
            
            # Execute queries in parallel
            results = []
            with ThreadPoolExecutor(max_workers=min(len(parsed_sections), 5)) as executor:
                future_to_section = {executor.submit(query_specific_section, section): section 
                                   for section in parsed_sections}
                
                for future in as_completed(future_to_section):
                    section = future_to_section[future]
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'model_type': section['model_type'],
                            'section_question': section['question'],
                            'success': False,
                            'error': str(e),
                            'answer': f"Timeout with {section['model_type']}: {str(e)}"
                        })
            
            # Sort results by the original order
            section_order = {section['model_type']: i for i, section in enumerate(parsed_sections)}
            results.sort(key=lambda x: section_order.get(x['model_type'], 999))
            
            # Create combined answer
            successful_results = [r for r in results if r.get('success', False)]
            
            combined_answer = f"# Targeted Multi-Model Response\n\n"
            
            for i, result in enumerate(results, 1):
                model_name = result['model_type'].title()
                section_q = result.get('section_question', '')
                
                combined_answer += f"## {i}. {model_name} Response\n"
                combined_answer += f"**Question**: \"{section_q}\"\n\n"
                
                if result.get('success', False):
                    combined_answer += f"{result['answer']}\n\n"
                else:
                    combined_answer += f"❌ **Error**: {result.get('error', 'Unknown error')}\n\n"
            
            # Collect all sources
            all_sources = []
            for result in successful_results:
                if result.get('sources'):
                    for source in result['sources']:
                        source_with_model = source.copy()
                        source_with_model['model_type'] = result['model_type']
                        source_with_model['section_question'] = result['section_question']
                        all_sources.append(source_with_model)
            
            # Create response
            response = {
                'question': question,
                'answer': combined_answer,
                'sources': all_sources,
                'source_type': 'parsed-sections',
                'prompt_type': 'auto-parsed-multi',
                'parsed_sections': parsed_sections,
                'model_types': [s['model_type'] for s in parsed_sections],
                'results': results,
                'successful_models': len(successful_results),
                'model_config': {'type': 'auto-parsed', 'sections': len(parsed_sections)},
                'timestamp': datetime.now().isoformat()
            }
            
            conversation_history.append(response)
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]
            
            return jsonify(response)
        
        # Regular single-model processing
        try:
            result = rag_instance.query_documents(question)
            
            # Check if the result indicates no relevant documents were found
            if result.get('source_type') == 'general_knowledge':
                # No documents found, automatically trigger multi-model mode for better coverage
                logger.info(f"No relevant documents found for question: '{question}'. Triggering automatic multi-model fallback.")
                
                try:
                    # Get available model types for multi-model fallback
                    from working_rag_app import MULTI_MODEL_CONFIG
                    available_models = list(MULTI_MODEL_CONFIG.keys())[:3]  # Use first 3 models to avoid overload
                    
                    logger.info(f"Auto-fallback using models: {available_models}")
                    
                    # Process with multiple models
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    
                    def query_with_model(model_type):
                        max_retries = 2
                        for attempt in range(max_retries):
                            try:
                                model_result = rag_instance.query_documents(question, force_prompt_type=model_type)
                                return {
                                    'model_type': model_type,
                                    'success': True,
                                    'answer': model_result['answer'],
                                    'sources': model_result.get('sources', []),
                                    'model_config': model_result.get('model_config', {}),
                                    'prompt_type': model_result.get('prompt_type', model_type)
                                }
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    logger.warning(f"Attempt {attempt + 1} failed for {model_type}: {str(e)}. Retrying...")
                                    import time
                                    time.sleep(1)
                                    continue
                                return {
                                    'model_type': model_type,
                                    'success': False,
                                    'error': str(e),
                                    'answer': f"❌ {model_type.title()} model unavailable: {str(e)}"
                                }
                    
                    # Execute queries in parallel
                    multi_results = []
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        future_to_model = {executor.submit(query_with_model, model): model 
                                         for model in available_models}
                        
                        for future in as_completed(future_to_model):
                            model = future_to_model[future]
                            try:
                                model_result = future.result(timeout=25)
                                multi_results.append(model_result)
                            except Exception as e:
                                multi_results.append({
                                    'model_type': model,
                                    'success': False,
                                    'error': str(e),
                                    'answer': f"⏰ {model.title()} model timeout: {str(e)}"
                                })
                    
                    # Sort results by model type for consistent display
                    multi_results.sort(key=lambda x: x['model_type'])
                    successful_results = [r for r in multi_results if r.get('success', False)]
                    
                    # Create combined multi-model answer
                    combined_answer = f"# Auto Multi-Model Response\n\n"
                    combined_answer += f"💡 **No relevant documents found** in your uploaded files for this question, so I'm using multiple AI models to provide comprehensive coverage:\n\n"
                    
                    for i, result in enumerate(multi_results, 1):
                        model_name = result['model_type'].title()
                        combined_answer += f"## {i}. {model_name} Model Response\n\n"
                        
                        if result.get('success', False):
                            combined_answer += f"{result['answer']}\n\n"
                        else:
                            combined_answer += f"❌ **Error**: {result.get('error', 'Unknown error')}\n\n"
                    
                    if successful_results:
                        combined_answer += f"\n---\n\n💭 **Note**: This response is generated from {len(successful_results)} different AI model{'s' if len(successful_results) > 1 else ''} since no relevant information was found in your document collection. Consider uploading relevant documents for more specific answers."
                    
                    # Collect all sources (though there likely won't be any)
                    all_sources = []
                    for result in successful_results:
                        if result.get('sources'):
                            for source in result['sources']:
                                source_with_model = source.copy()
                                source_with_model['model_type'] = result['model_type']
                                all_sources.append(source_with_model)
                    
                    # Create enhanced response indicating auto multi-model fallback
                    response = {
                        'question': question,
                        'answer': combined_answer,
                        'sources': all_sources,
                        'source_type': 'auto_multi_model_fallback',
                        'prompt_type': 'auto-multi-fallback',
                        'auto_fallback_triggered': True,
                        'fallback_reason': 'No relevant documents found',
                        'used_models': available_models,
                        'successful_models': len(successful_results),
                        'multi_results': multi_results,
                        'model_config': {'type': 'auto-fallback', 'models_used': len(available_models)},
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    conversation_history.append(response)
                    if len(conversation_history) > 50:
                        conversation_history = conversation_history[-50:]
                    
                    logger.info(f"Auto multi-model fallback completed. Used {len(successful_results)}/{len(available_models)} models successfully.")
                    return jsonify(response)
                
                except Exception as e:
                    logger.error(f"Auto multi-model fallback failed: {str(e)}")
                    # Fall through to return the original single-model result
            
            # Create response for single model (either with documents or as fallback)
            response = {
                'question': question,
                'answer': result['answer'],
                'sources': result.get('sources', []),
                'source_type': result.get('source_type', 'documents'),
                'prompt_type': result.get('prompt_type', 'default'),
                'model_config': result.get('model_config', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to conversation history
            conversation_history.append(response)
            
            # Keep only last 50 conversations
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]
            
            return jsonify(response)
        
        except Exception as e:
            error_msg = str(e)
            
            # Provide user-friendly error messages based on error type
            if "timeout" in error_msg.lower():
                user_message = "⏰ **Connection Timeout**: The AI model server is taking longer than expected to respond. This might be due to high server load or network issues.\n\n**Suggestions:**\n• Try asking your question again\n• Simplify your question if it's very complex\n• Check your internet connection"
            elif "404" in error_msg or "not found" in error_msg.lower():
                user_message = "🔍 **Service Unavailable**: The AI model service could not be reached. The requested model may not be available.\n\n**Suggestions:**\n• Try again in a few moments\n• Contact your administrator if this persists"
            elif "connection" in error_msg.lower():
                user_message = "🌐 **Connection Error**: Unable to connect to the AI model server.\n\n**Suggestions:**\n• Check your internet connection\n• Verify that the server is accessible\n• Try again in a few moments"
            else:
                user_message = f"❌ **Service Error**: There was an issue processing your request.\n\nError details: {error_msg}\n\n**Suggestions:**\n• Try rephrasing your question\n• Contact support if the issue continues"
            
            # Create error response that looks like a regular response
            error_response = {
                'question': question,
                'answer': user_message,
                'sources': [],
                'source_type': 'error',
                'prompt_type': 'error',
                'model_config': {'error': True, 'error_type': type(e).__name__},
                'timestamp': datetime.now().isoformat(),
                'error_details': error_msg
            }
            
            # Still add to history for troubleshooting
            conversation_history.append(error_response)
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]
            
            return jsonify(error_response)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask-multi', methods=['POST'])
def ask_multi_model():
    """Process question using multiple specified models in parallel"""
    global conversation_history
    
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        model_types = data.get('model_types', [])
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if not model_types or not isinstance(model_types, list):
            return jsonify({'error': 'model_types must be a non-empty list'}), 400
        
        # Validate model types exist
        from working_rag_app import MULTI_MODEL_CONFIG
        all_available_types = set(MULTI_MODEL_CONFIG.keys()) | set(rag_instance.custom_prompt_types.keys())
        
        invalid_types = [t for t in model_types if t not in all_available_types]
        if invalid_types:
            return jsonify({'error': f'Invalid model types: {invalid_types}'}), 400
        
        # Query each model in parallel using threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        def query_single_model(model_type):
            try:
                # Create a thread-local copy for thread safety
                thread_local_question = f"{question}"
                result = rag_instance.query_documents(thread_local_question, force_prompt_type=model_type)
                return {
                    'model_type': model_type,
                    'success': True,
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'model_config': result.get('model_config', {}),
                    'prompt_type': result.get('prompt_type', model_type)
                }
            except Exception as e:
                return {
                    'model_type': model_type,
                    'success': False,
                    'error': str(e),
                    'answer': f"Error processing with {model_type}: {str(e)}"
                }
        
        # Execute queries in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(len(model_types), 5)) as executor:
            future_to_type = {executor.submit(query_single_model, model_type): model_type 
                             for model_type in model_types}
            
            for future in as_completed(future_to_type):
                model_type = future_to_type[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per model
                    results.append(result)
                except Exception as e:
                    results.append({
                        'model_type': model_type,
                        'success': False,
                        'error': str(e),
                        'answer': f"Timeout or error with {model_type}: {str(e)}"
                    })
        
        # Sort results by the original order of model_types
        results.sort(key=lambda x: model_types.index(x['model_type']))
        
        # Combine successful results
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        # Create combined answer
        if successful_results:
            combined_answer = f"# Multi-Model Response\n\nQuestion: {question}\n\n"
            
            for i, result in enumerate(results, 1):
                model_name = result['model_type'].title()
                if result.get('success', False):
                    combined_answer += f"## {i}. {model_name} Perspective:\n{result['answer']}\n\n"
                else:
                    combined_answer += f"## {i}. {model_name} Perspective:\n❌ Error: {result.get('error', 'Unknown error')}\n\n"
            
            # Add summary if multiple successful results
            if len(successful_results) > 1:
                combined_answer += "## Summary:\n"
                combined_answer += f"Successfully received {len(successful_results)} perspectives from different AI models. "
                combined_answer += "Each model brings its specialized knowledge and approach to provide comprehensive insights.\n"
        else:
            combined_answer = f"❌ All models failed to process the question: {question}\n\n"
            for result in failed_results:
                combined_answer += f"- {result['model_type']}: {result.get('error', 'Unknown error')}\n"
        
        # Collect all unique sources
        all_sources = []
        for result in successful_results:
            if result.get('sources'):
                for source in result['sources']:
                    # Add model type to source for identification
                    source_with_model = source.copy()
                    source_with_model['model_type'] = result['model_type']
                    all_sources.append(source_with_model)
        
        # Create response
        response = {
            'question': question,
            'answer': combined_answer,
            'sources': all_sources,
            'source_type': 'multi-model',
            'prompt_type': 'multi-model',
            'model_types': model_types,
            'results': results,
            'successful_models': len(successful_results),
            'failed_models': len(failed_results),
            'model_config': {'type': 'multi-model', 'models': model_types},
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to conversation history
        conversation_history.append(response)
        
        # Keep only last 50 conversations
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing multi-model question: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask-parsed-multi', methods=['POST'])
def ask_parsed_multi_model():
    """Process question with model-specific sections parsed from the text"""
    global conversation_history
    
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Parse the question for model-specific sections
        parsed_sections = parse_model_specific_question(question)
        
        if not parsed_sections:
            # No specific model sections found, use regular single model processing
            result = rag_instance.query_documents(question)
            response = {
                'question': question,
                'answer': result['answer'],
                'sources': result.get('sources', []),
                'source_type': result.get('source_type', 'documents'),
                'prompt_type': result.get('prompt_type', 'default'),
                'model_config': result.get('model_config', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            conversation_history.append(response)
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-50:]
            
            return jsonify(response)
        
        # Validate that all specified model types exist
        from working_rag_app import MULTI_MODEL_CONFIG
        all_available_types = set(MULTI_MODEL_CONFIG.keys()) | set(rag_instance.custom_prompt_types.keys())
        
        invalid_types = [section['model_type'] for section in parsed_sections 
                        if section['model_type'] not in all_available_types]
        if invalid_types:
            return jsonify({'error': f'Invalid model types: {invalid_types}'}), 400
        
        # Query each model with its specific question section
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def query_specific_section(section):
            try:
                model_type = section['model_type']
                section_question = section['question']
                result = rag_instance.query_documents(section_question, force_prompt_type=model_type)
                return {
                    'model_type': model_type,
                    'section_question': section_question,
                    'success': True,
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'model_config': result.get('model_config', {}),
                    'prompt_type': result.get('prompt_type', model_type)
                }
            except Exception as e:
                return {
                    'model_type': section['model_type'],
                    'section_question': section['question'],
                    'success': False,
                    'error': str(e),
                    'answer': f"Error processing {section['model_type']} section: {str(e)}"
                }
        
        # Execute queries in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(len(parsed_sections), 5)) as executor:
            future_to_section = {executor.submit(query_specific_section, section): section 
                               for section in parsed_sections}
            
            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per model
                    results.append(result)
                except Exception as e:
                    results.append({
                        'model_type': section['model_type'],
                        'section_question': section['question'],
                        'success': False,
                        'error': str(e),
                        'answer': f"Timeout with {section['model_type']}: {str(e)}"
                    })
        
        # Sort results by the original order of sections
        section_order = {section['model_type']: i for i, section in enumerate(parsed_sections)}
        results.sort(key=lambda x: section_order.get(x['model_type'], 999))
        
        # Combine results by section
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        # Create combined answer with sections
        if successful_results or failed_results:
            combined_answer = f"# Multi-Model Response with Targeted Sections\n\n"
            combined_answer += f"**Original Question**: {question}\n\n"
            
            for i, result in enumerate(results, 1):
                model_name = result['model_type'].title()
                section_q = result.get('section_question', 'Unknown')
                
                combined_answer += f"## {i}. {model_name} Response\n"
                combined_answer += f"**Section**: \"{section_q}\"\n\n"
                
                if result.get('success', False):
                    combined_answer += f"{result['answer']}\n\n"
                else:
                    combined_answer += f"❌ **Error**: {result.get('error', 'Unknown error')}\n\n"
            
            # Add summary
            if len(successful_results) > 1:
                combined_answer += "## Summary\n"
                combined_answer += f"Successfully processed {len(successful_results)} targeted sections using specialized AI models. "
                combined_answer += "Each model focused on its specific part of your question for optimal results.\n"
        else:
            combined_answer = f"❌ All sections failed to process: {question}\n\n"
            for result in failed_results:
                combined_answer += f"- {result['model_type']}: {result.get('error', 'Unknown error')}\n"
        
        # Collect all unique sources
        all_sources = []
        for result in successful_results:
            if result.get('sources'):
                for source in result['sources']:
                    source_with_model = source.copy()
                    source_with_model['model_type'] = result['model_type']
                    source_with_model['section_question'] = result['section_question']
                    all_sources.append(source_with_model)
        
        # Create response
        response = {
            'question': question,
            'answer': combined_answer,
            'sources': all_sources,
            'source_type': 'parsed-multi-model',
            'prompt_type': 'parsed-multi-model',
            'parsed_sections': parsed_sections,
            'model_types': [s['model_type'] for s in parsed_sections],
            'results': results,
            'successful_models': len(successful_results),
            'failed_models': len(failed_results),
            'model_config': {'type': 'parsed-multi-model', 'sections': len(parsed_sections)},
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to conversation history
        conversation_history.append(response)
        
        # Keep only last 50 conversations
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-50:]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing parsed multi-model question: {e}")
        return jsonify({'error': str(e)}), 500

def parse_model_specific_question(question):
    """
    Parse a question for model-specific sections using patterns like:
    'Technical: Car engine details, Creative: how to make dream car'
    
    Returns list of {'model_type': str, 'question': str} dictionaries
    """
    import re
    from working_rag_app import MULTI_MODEL_CONFIG
    
    # Get all available model types (case insensitive)
    available_types = set(MULTI_MODEL_CONFIG.keys())
    if hasattr(rag_instance, 'custom_prompt_types'):
        available_types.update(rag_instance.custom_prompt_types.keys())
    
    # Create pattern that matches any available model type followed by colon
    type_pattern = r'\b(' + '|'.join(re.escape(t) for t in available_types) + r')\s*:\s*'
    
    # Find all sections with case-insensitive matching
    sections = []
    last_end = 0
    
    for match in re.finditer(type_pattern, question, re.IGNORECASE):
        model_type = match.group(1).lower()
        start = match.end()
        
        # Find the next model type or end of string
        next_match = None
        for next_m in re.finditer(type_pattern, question[start:], re.IGNORECASE):
            next_match = next_m
            break
        
        if next_match:
            end = start + next_match.start()
            section_text = question[start:end].strip()
        else:
            section_text = question[start:].strip()
        
        # Clean up section text (remove trailing commas, periods, etc.)
        section_text = re.sub(r'[,;.]*$', '', section_text).strip()
        
        if section_text:
            sections.append({
                'model_type': model_type,
                'question': section_text
            })
    
    return sections

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    limit = request.args.get('limit', 20, type=int)
    return jsonify({
        'history': conversation_history[-limit:],
        'total': len(conversation_history)
    })

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return jsonify({'message': 'History cleared'})

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """Upload and process documents for RAG"""
    if not rag_instance:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        processed_files = []
        all_docs = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Process the uploaded file
                try:
                    if filename.lower().endswith('.pdf'):
                        docs = rag_instance.load_pdf_documents([file_path])
                        all_docs.extend(docs)
                        processed_files.append({
                            'filename': filename,
                            'type': 'PDF',
                            'chunks': len(docs),
                            'status': 'success'
                        })
                    elif filename.lower().endswith('.txt'):
                        # Process text files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Split text into chunks
                        chunk_size = 1000
                        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                        
                        docs = []
                        for i, chunk in enumerate(chunks):
                            doc = SimpleDocument(
                                content=chunk,
                                metadata={
                                    'source_file': filename,
                                    'chunk_id': i,
                                    'file_type': 'text'
                                }
                            )
                            docs.append(doc)
                        
                        all_docs.extend(docs)
                        processed_files.append({
                            'filename': filename,
                            'type': 'Text',
                            'chunks': len(docs),
                            'status': 'success'
                        })
                    else:
                        processed_files.append({
                            'filename': filename,
                            'type': 'Unsupported',
                            'chunks': 0,
                            'status': 'error',
                            'error': 'File type not supported'
                        })
                
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    processed_files.append({
                        'filename': filename,
                        'type': 'Error',
                        'chunks': 0,
                        'status': 'error',
                        'error': str(e)
                    })
                
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass
            else:
                processed_files.append({
                    'filename': file.filename,
                    'type': 'Invalid',
                    'chunks': 0,
                    'status': 'error',
                    'error': 'Invalid file type'
                })
        
        # Add documents to vector store
        if all_docs:
            rag_instance.process_documents(all_docs)
            rag_instance.save_vectorstore()
            
            return jsonify({
                'message': f'Successfully processed {len(all_docs)} document chunks',
                'files': processed_files,
                'total_chunks': len(all_docs)
            })
        else:
            return jsonify({
                'message': 'No valid documents were processed',
                'files': processed_files,
                'total_chunks': 0
            }), 400
            
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get information about loaded documents"""
    if not rag_instance or not rag_instance.vectorstore.documents:
        return jsonify({'documents': [], 'total': 0})
    
    doc_info = []
    sources = set()
    
    for doc in rag_instance.vectorstore.documents:
        source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
        sources.add(source)
        
        doc_info.append({
            'source': source,
            'content_preview': doc.page_content[:100] + '...',
            'metadata': doc.metadata
        })
    
    return jsonify({
        'documents': doc_info[:20],  # Return first 20 for performance
        'total': len(doc_info),
        'sources': list(sources),
        'source_count': len(sources)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize RAG system
    if initialize_rag():
        print("🚀 Starting Flask server...")
        print("📱 Mobile UI: http://localhost:5000/mobile")
        print("💻 Web UI: http://localhost:5000/")
        print("🔗 API: http://localhost:5000/api/")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize RAG system")
