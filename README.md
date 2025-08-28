# 🚀 Ollama Chat App with RAG - Production Ready

A high-performance, scalable chat application powered by Ollama with advanced RAG (Retrieval-Augmented Generation) capabilities, professional prompt engineering, and intelligent caching system.

## ✨ Key Features

### 🧠 Advanced AI Capabilities
- **Professional Prompt Engineering**: Intelligent prompt optimization for different use cases (coding, business, creative)
- **Smart Response Caching**: Ultra-fast responses with intelligent cache management (>1000x speed improvement)
- **Multi-model Support**: Chat with any Ollama model (qwen3, gpt-oss, mistral, codellama, etc.)
- **Context-Aware Sessions**: Persistent conversations with session management and history

### 📚 Powerful RAG System
- **Document Processing**: Support for PDF, DOCX, TXT, CSV, XLSX files with intelligent text extraction
- **Semantic Search**: Advanced document search using `nomic-embed-text:v1.5` embeddings
- **Context Retrieval**: AI responses enhanced with relevant document context
- **Multiple Query Modes**: RAG, Search, Basic Chat, and Summarization modes
- **Real-time File Upload**: Drag-and-drop interface with progress tracking

### 🚀 Performance & Scalability
- **Intelligent Caching**: LRU cache with configurable TTL for lightning-fast responses
- **Async Architecture**: Non-blocking operations with aiofiles and async processing
- **Rate Limiting**: Configurable global rate limit per client (300 req/min default)
- **Session Management**: TTL-based cleanup with configurable limits
- **Background Tasks**: Automatic cleanup and maintenance

### 🔒 Production Security
- **CORS Protection**: Environment-based CORS configuration
- **Host Validation**: TrustedHostMiddleware for production deployments
- **Input Validation**: Comprehensive Pydantic validation with length limits
- **Security Headers**: XSS protection, content type validation, frame options
- **Environment Isolation**: Development, staging, and production configurations

### 📊 Monitoring & Observability
- **Prometheus Metrics**: HTTP requests, cache performance, session counts, Ollama API calls
- **Structured Logging**: JSON-formatted logs with request tracing
- **Health Checks**: Comprehensive health endpoint with system metrics
- **Performance Tracking**: Request timing and response generation metrics

### 🎨 Modern Interface
- **ChatGPT-like UI**: Three-panel layout with chat, sidebar, and RAG panel
- **Markdown Support**: Professional rendering with code syntax highlighting
- **Real-time Updates**: Live document statistics and cache status
- **Mobile Responsive**: Works seamlessly on all devices

## 📋 Requirements

- **Python 3.10+**
- **Ollama** (running locally or remotely)
- **Redis** (optional, for production session storage)
- **Docker & Docker Compose** (for containerized deployment)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd ollama-chat-app
python3 -m venv env-chat
source env-chat/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama
```bash
# Install and start Ollama
ollama serve

# Pull required models
ollama pull qwen3:latest
ollama pull nomic-embed-text:v1.5
```

### 3. Configure Environment
```bash
# Create .env file (optional)
echo "ENVIRONMENT=development" > .env
echo "CORS_ORIGINS=*" >> .env
echo "RATE_LIMIT_REQUESTS=300" >> .env
```

### 4. Run the Application
```bash
# Development mode
python app.py

# Or with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Application
- **Web Interface**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/health
- **API Documentation**: http://localhost:8000/docs

## 📚 Usage Guide

### Basic Chat
1. Open http://localhost:8000
2. Select a model from the dropdown
3. Start chatting with professional AI responses

### RAG (Document Chat)
1. Click "RAG" mode in the interface
2. Upload documents (PDF, DOCX, TXT, CSV, XLSX)
3. Ask questions about your documents
4. Get AI responses with document context

### API Usage
```bash
# Health check
curl http://localhost:8000/api/health

# Chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello","model":"qwen3:latest","temperature":0.2}'

# Upload document
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@document.pdf"

# RAG query
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is this document about?","model":"qwen3:latest"}'
```

## 🔧 Configuration

### Environment Variables
```bash
# Application
ENVIRONMENT=development  # development|staging|production
DEBUG=true
APP_NAME="Ollama Chat App with RAG"

# Server
HOST=0.0.0.0
PORT=8000

# Security
CORS_ORIGINS=*  # or specific domains
ALLOWED_HOSTS=*  # or specific hosts
RATE_LIMIT_REQUESTS=300

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30

# Caching
ENABLE_CACHING=true
CACHE_TTL=300  # 5 minutes

# File Upload
MAX_UPLOAD_SIZE=52428800  # 50MB
UPLOAD_DIR=uploads

# RAG
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_EMBEDDING_MODEL=nomic-embed-text:v1.5
```

### Production Configuration
```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
CORS_ORIGINS=https://yourdomain.com
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
RATE_LIMIT_REQUESTS=100

# Database (optional)
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/ollama_chat
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build
```bash
# Build image
docker build -t ollama-chat-app .

# Run container
docker run -d \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ollama-chat-app
```

## 📊 Monitoring

### Health Endpoint
```bash
curl http://localhost:8000/api/health
```

Response includes:
- Application status
- Ollama connectivity
- System metrics (CPU, memory, disk)
- Cache status
- Session counts

### Prometheus Metrics
Access metrics at `http://localhost:8000/metrics`:
- `http_requests_total` - Total HTTP requests
- `cache_hits_total` - Cache hit rate
- `active_sessions` - Current active sessions
- `response_generation_seconds` - AI response times

### Logs
Structured JSON logs include:
- Request tracing
- Performance metrics
- Error tracking
- Cache operations

## 🔧 Advanced Features

### Professional Prompts
The app includes intelligent prompt engineering:
- **Coding Mode**: Optimized for programming tasks
- **Business Mode**: Professional communication
- **Creative Mode**: Enhanced creativity
- **RAG Mode**: Document-focused responses

### Caching System
- **LRU Cache**: Least Recently Used eviction
- **TTL Support**: Time-based expiration
- **Cache Statistics**: Hit/miss tracking
- **Performance Gains**: >1000x speed improvement for cached responses

### Session Management
- **User Isolation**: Separate sessions per user
- **TTL Cleanup**: Automatic session expiration
- **Session Limits**: Configurable per-user limits
- **History Persistence**: Chat history retention

## 🛠️ Development

### Run Tests
```bash
# Test configuration
python config.py

# Test all imports
python -c "from app import app; print('✅ All imports working')"

# Test endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/api/models
```

### Development Mode
```bash
# Start with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## 📁 Project Structure

```
ollama-chat-app/
├── app.py                      # Main FastAPI application
├── config.py                   # Configuration management
├── professional_prompts.py     # Prompt engineering system
├── rag_service.py              # RAG implementation
├── file_service.py             # File handling service
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-service deployment
├── nginx.conf                  # Reverse proxy configuration
├── start_production.sh         # Production startup script
├── static/
│   ├── index.html              # Frontend interface
│   └── favicon.ico             # App icon
├── uploads/                    # Document storage
│   ├── pdf/
│   ├── docx/
│   ├── txt/
│   ├── csv/
│   └── xlsx/
└── env-chat/                   # Virtual environment
```

## 🚨 Troubleshooting

### Common Issues

1. **Port 8000 already in use**
   ```bash
   pkill -f "python.*app.py"
   # or change port
   export PORT=8001
   ```

2. **Ollama not connected**
   ```bash
   ollama serve
   curl http://localhost:11434/api/version
   ```

3. **Missing models**
   ```bash
   ollama pull qwen3:latest
   ollama pull nomic-embed-text:v1.5
   ```

4. **Cache issues**
   ```bash
   # Clear cache
   curl -X POST http://localhost:8000/api/cache/clear
   ```

5. **Permission errors**
   ```bash
   chmod +x start_production.sh
   mkdir -p uploads logs
   ```

### Debug Mode
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** - Local AI model serving
- **FastAPI** - Modern web framework
- **LangChain** - RAG implementation framework
- **Pydantic** - Data validation and settings
- **Prometheus** - Metrics and monitoring

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error details
3. Open an issue on GitHub
4. Include system info and logs

---

**Built with ❤️ for the AI community**