# 🚀 Ollama Chat App with RAG - Production Ready

A production-ready, scalable chat application powered by Ollama with advanced RAG (Retrieval-Augmented Generation) capabilities for document processing and intelligent querying.

## ✨ Features

### 🤖 Core Chat Functionality
- **Professional AI Responses**: Clean, direct responses without "thinking out loud" language
- **Multi-model Support**: Chat with any Ollama model (llama3.2, gpt-oss, mistral, codellama, etc.)
- **Advanced Session Management**: Multi-user support with session limits and TTL
- **Real-time Streaming**: Fast, responsive chat interface
- **Model Switching**: Seamlessly switch between different AI models

### 📚 Advanced RAG (Retrieval-Augmented Generation)
- **Document Upload**: Support for PDF, DOCX, TXT, CSV, XLSX files
- **Intelligent Chunking**: Smart text segmentation with configurable chunk sizes
- **Semantic Search**: Find relevant information using `nomic-embed-text:v1.5` embeddings
- **Context-Aware Responses**: AI answers based on uploaded document content
- **Multiple Modes**: RAG, Search, Basic Chat, and Summarization modes

### 🏗️ Production Features
- **Production Ready**: Rate limiting, monitoring, logging, and security headers
- **Scalable Architecture**: Support for Redis, multiple workers, and load balancing
- **Modern UI**: Responsive ChatGPT-like interface with markdown support
- **Health Monitoring**: Prometheus metrics and comprehensive health checks

## 📋 Requirements

- Python 3.11+
- Ollama (running locally or remotely)
- Redis (for production session storage)
- Docker & Docker Compose (for containerized deployment)

## 🛠️ Installation

### Option 1: Direct Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ollama-chat-app
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv env-chat
   source env-chat/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama**
   ```bash
   ollama serve
   ```

5. **Install Ollama models**
   ```bash
   ollama pull nomic-embed-text:v1.5
   ollama pull gpt-oss:20B  # or any other model you prefer
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:8000`

## 📖 RAG Usage Guide

### 1. Document Upload
- Navigate to the RAG panel on the right side
- Click "Upload File(s)" or drag & drop files
- Supported formats: PDF, DOCX, TXT, CSV, XLSX, XLS
- Maximum file sizes: PDF (50MB), DOCX/XLSX (20MB), TXT/CSV (10MB)

### 2. RAG Queries
- **RAG Mode**: Ask questions about uploaded documents
- **Search Mode**: Find specific information in documents
- **Basic Mode**: Regular chat without document context
- **Summarize Mode**: Get document summaries

### 3. Example RAG Workflow
```bash
# 1. Upload a document
curl -X POST -F "file=@document.pdf" http://localhost:8000/api/rag/upload

# 2. Query the document
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What are the main points?", "model": "gpt-oss:20B"}' \
  http://localhost:8000/api/rag/query

# 3. Search for specific content
curl "http://localhost:8000/api/rag/search?query=artificial%20intelligence&top_k=5"
```

## 📊 API Endpoints

### Chat Endpoints
- `POST /api/chat` - Send chat message
- `POST /api/sessions/new` - Create new session
- `GET /api/sessions` - List all sessions

### RAG Endpoints
- `POST /api/rag/upload` - Upload document
- `POST /api/rag/query` - Query documents with RAG
- `GET /api/rag/documents` - List uploaded documents
- `DELETE /api/rag/documents/{id}` - Delete document
- `GET /api/rag/stats` - Get RAG statistics
- `GET /api/rag/search` - Search documents

### System Endpoints
- `GET /api/health` - Health check
- `GET /metrics` - Prometheus metrics

### Option 2: Production Deployment with Docker

1. **Build and start services**
   ```bash
   docker-compose up -d
   ```

2. **Check status**
   ```bash
   docker-compose ps
   ```

3. **View logs**
   ```bash
   docker-compose logs -f ollama-chat-app
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Application
ENVIRONMENT=production
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30
OLLAMA_MAX_RETRIES=3

# RAG Configuration
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_EMBEDDING_MODEL=nomic-embed-text:v1.5

# Session Management
MAX_SESSIONS_PER_USER=100
MAX_MESSIONS_PER_SESSION=1000
SESSION_CLEANUP_INTERVAL=3600
SESSION_TTL=604800

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Security
CORS_ORIGINS=https://yourdomain.com,http://localhost:3000
ENABLE_API_KEY_AUTH=false

# Database
REDIS_URL=redis://localhost:6379

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001
```

### Production Scripts

Use the provided production scripts for easy management:

```bash
# Start the application
./start_production.sh start

# Check status
./start_production.sh status

# View logs
./start_production.sh logs

# Stop the application
./start_production.sh stop

# Restart the application
./start_production.sh restart
```

## 📊 Monitoring & Metrics

### Health Check

```bash
curl http://localhost:8000/api/health
```

Response includes:
- Application status
- Ollama connection status
- Session counts
- System metrics (CPU, memory, disk)

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Available metrics:
- `http_requests_total`: Total HTTP requests by method, endpoint, and status
- `http_request_duration_seconds`: Request duration histogram
- `active_sessions`: Number of active sessions
- `total_sessions`: Total number of sessions
- `ollama_requests_total`: Total Ollama API requests
- `ollama_errors_total`: Total Ollama API errors

### Logging

The application uses structured logging with JSON format. Logs include:
- Request/response details
- Error tracking
- Performance metrics
- Session management events

## 🔒 Security Features

- **Rate Limiting**: Configurable per-endpoint rate limits
- **CORS Protection**: Configurable cross-origin policies
- **Security Headers**: XSS protection, content type validation, frame options
- **Input Validation**: Comprehensive request validation with Pydantic
- **Session Isolation**: User session isolation and access control

## 📈 Scalability Features

- **Multiple Workers**: Support for multiple Gunicorn workers
- **Redis Integration**: Scalable session storage (optional)
- **Load Balancing**: Ready for Nginx load balancer
- **Horizontal Scaling**: Containerized deployment support
- **Resource Management**: Configurable limits and cleanup

## 🚨 Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if not running
   ollama serve
   ```

2. **Port Already in Use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   
   # Kill the process
   kill -9 <PID>
   ```

3. **Memory Issues**
   ```bash
   # Check memory usage
   free -h
   
   # Reduce worker count in config.py
   WORKERS=2
   ```

4. **Rate Limit Exceeded**
   - Increase `RATE_LIMIT_REQUESTS` in configuration
   - Implement client-side retry logic
   - Consider using multiple API keys

### Performance Tuning

1. **Worker Configuration**
   ```bash
   # For CPU-bound workloads
   WORKERS = CPU_CORES * 2 + 1
   
   # For I/O-bound workloads
   WORKERS = CPU_CORES * 4
   ```

2. **Session Cleanup**
   ```bash
   # More frequent cleanup for high-traffic apps
   SESSION_CLEANUP_INTERVAL=1800  # 30 minutes
   ```

3. **Rate Limiting**
   ```bash
   # Adjust based on your Ollama model performance
   RATE_LIMIT_REQUESTS=50  # More conservative
   ```

## 🔄 Deployment

### Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Configure `CORS_ORIGINS` with your domain
- [ ] Set up SSL certificates
- [ ] Configure Redis for session storage
- [ ] Set up monitoring and alerting
- [ ] Configure log rotation
- [ ] Set up backup procedures
- [ ] Test rate limiting and security

### SSL Setup

1. **Generate self-signed certificate (development)**
   ```bash
   mkdir ssl
   openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
   ```

2. **Use Let's Encrypt (production)**
   ```bash
   certbot certonly --standalone -d yourdomain.com
   ```

### Nginx Configuration

The provided `nginx.conf` includes:
- SSL termination
- Rate limiting
- Security headers
- Gzip compression
- Static file serving
- Load balancing

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs: `./start_production.sh logs`
3. Check the health endpoint
4. Open an issue on GitHub

## 🔮 Roadmap

- [ ] Database persistence (PostgreSQL)
- [ ] User authentication and authorization
- [ ] Multi-tenant support
- [ ] Advanced analytics dashboard
- [ ] WebSocket support for real-time chat
- [ ] Model fine-tuning interface
- [ ] API key management
- [ ] Advanced rate limiting strategies
