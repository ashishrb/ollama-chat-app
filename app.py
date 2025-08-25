from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import ollama
import json
import os
import uuid
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
import re
import logging
import structlog
from collections import defaultdict, deque
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import RAG services
from rag_service import RAGService
from file_service import FileService

# Import settings
try:
    from config import settings
except ImportError:
    # Fallback settings if config.py doesn't exist
    class Settings:
        debug = os.getenv("DEBUG", "false").lower() == "true"
        app_name = "Ollama Chat App with RAG"
        app_version = "1.0.0"
        cors_origins = ["*"]
        max_sessions_per_user = 100
        max_messages_per_session = 1000
        session_ttl = 86400 * 7
        session_cleanup_interval = 3600
        rate_limit_requests = 100
        rate_limit_window = 60
        ollama_timeout = 30
        ollama_max_retries = 3
    
    settings = Settings()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_SESSIONS = Gauge('active_sessions', 'Number of active sessions')
TOTAL_SESSIONS = Gauge('total_sessions', 'Total number of sessions')
OLLAMA_REQUESTS = Counter('ollama_requests_total', 'Total Ollama API requests')
OLLAMA_ERRORS = Counter('ollama_errors_total', 'Total Ollama API errors')

# Global storage with TTL support
chat_sessions: Dict[str, Dict[str, Any]] = {}
active_sessions: Dict[str, str] = {}
session_creation_times: Dict[str, float] = {}
rate_limit_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

# Rate limiting
def check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit"""
    now = time.time()
    client_requests = rate_limit_store[client_id]
    
    # Remove old requests outside the window
    while client_requests and now - client_requests[0] > settings.rate_limit_window:
        client_requests.popleft()
    
    # Check if limit exceeded
    if len(client_requests) >= settings.rate_limit_requests:
        return False
    
    # Add current request
    client_requests.append(now)
    return True

def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting"""
    try:
        # Use X-Forwarded-For for proxy support, fallback to client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Safely get client host
        if request.client and hasattr(request.client, 'host'):
            return request.client.host
        
        # Fallback to request headers
        host = request.headers.get("Host", "unknown")
        return host.split(":")[0] if ":" in host else host
        
    except Exception:
        return "unknown"

# Session cleanup
async def cleanup_expired_sessions():
    """Remove expired sessions"""
    now = time.time()
    expired_sessions = []
    
    for session_id, creation_time in session_creation_times.items():
        if now - creation_time > settings.session_ttl:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        if session_id in session_creation_times:
            del session_creation_times[session_id]
        
        # Remove from active sessions
        for user_id, active_session_id in list(active_sessions.items()):
            if active_session_id == session_id:
                del active_sessions[user_id]
    
    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        TOTAL_SESSIONS.set(len(chat_sessions))
        ACTIVE_SESSIONS.set(len(active_sessions))

# Pydantic models with validation
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: str
    message_id: str
    usage: Optional[int] = None
    timestamp: str

class SessionInfo(BaseModel):
    session_id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    message_count: int

class NewSessionRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = Field(None, max_length=200)

# RAG Models
class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_chunks: int = Field(3, ge=1, le=10)
    include_context: bool = Field(True)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class RAGQueryResponse(BaseModel):
    response: str
    query: str
    model: str
    context_used: List[Dict[str, Any]]
    timestamp: str

class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    file_name: str
    chunk_count: int
    total_length: int
    message: str

class DocumentInfo(BaseModel):
    id: str
    name: str
    extension: str
    chunk_count: int
    total_length: int
    uploaded_at: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Ollama Chat App", version=settings.app_version)
    
    # Initialize RAG and File services
    global rag_service, file_service
    rag_service = RAGService()
    file_service = FileService()
    
    try:
        # Test Ollama connection
        ollama.list()
        logger.info("Ollama connection successful")
        
        # Initialize RAG service
        await rag_service.initialize()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error("Ollama connection failed", error=str(e))
    
    # Start background tasks
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ollama Chat App")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

async def periodic_cleanup():
    """Periodically clean up expired sessions"""
    while True:
        try:
            await cleanup_expired_sessions()
            await asyncio.sleep(settings.session_cleanup_interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))
            await asyncio.sleep(60)  # Wait 1 minute before retrying

app = FastAPI(
    title=settings.app_name,
    description="Advanced chat application with conversation management",
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug
)

# Add production middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
# Only add TrustedHostMiddleware in production, not in development
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["localhost", "127.0.0.1", "localhost:8000", "127.0.0.1:8000", "0.0.0.0:8000", "yourdomain.com"]
    )

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Middleware for request timing and metrics
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    try:
        # Check rate limit (only for API endpoints, not static files or root)
        if (not request.url.path.startswith("/static") and 
            not request.url.path in ["/", "/favicon.ico"] and
            request.url.path.startswith("/api/")):
            try:
                client_id = get_client_id(request)
                if not check_rate_limit(client_id):
                    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=429).inc()
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded. Please try again later."}
                    )
            except Exception as e:
                logger.warning(f"Rate limiting failed for {request.url.path}: {str(e)}")
                # Continue without rate limiting if it fails
        
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
        REQUEST_DURATION.observe(process_time)
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}")
        # Continue with request even if middleware fails
        response = await call_next(request)
        return response

def get_or_create_user_id() -> str:
    """Generate or retrieve user ID (in production, implement proper user authentication)"""
    return str(uuid.uuid4())

def create_new_session(model: str, title: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    default_title = title or f"New Chat - {datetime.now().strftime('%H:%M')}"
    
    session = {
        "session_id": session_id,
        "title": default_title,
        "model": model,
        "user_id": user_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": [],
        "message_count": 0
    }
    
    chat_sessions[session_id] = session
    session_creation_times[session_id] = time.time()
    
    # Update metrics
    TOTAL_SESSIONS.set(len(chat_sessions))
    ACTIVE_SESSIONS.set(len(active_sessions))
    
    logger.info("Created new session", session_id=session_id, model=model, user_id=user_id)
    return session

def update_session_title(session_id: str, message: str) -> None:
    """Update session title based on first user message"""
    if session_id in chat_sessions:
        session = chat_sessions[session_id]
        if session["message_count"] == 1:  # First message
            # Generate a title from the first message
            words = message.split()[:5]  # First 5 words
            title = " ".join(words) + ("..." if len(message.split()) > 5 else "")
            session["title"] = title
            session["updated_at"] = datetime.now().isoformat()

def add_message_to_session(session_id: str, role: str, content: str, model: str) -> str:
    """Add a message to a session and return message ID"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    message_id = str(uuid.uuid4())
    
    message = {
        "id": message_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "model": model
    }
    
    session["messages"].append(message)
    session["message_count"] = len(session["messages"])
    session["updated_at"] = datetime.now().isoformat()
    
    return message_id

def clean_ai_response(content: str) -> str:
    """Clean AI response to remove thinking patterns and improve quality"""
    # Remove common thinking patterns and unprofessional language
    thinking_patterns = [
        r'Okay, I need to.*?\.',
        r'Let me think.*?\.',
        r'First, I remember.*?\.',
        r'But wait.*?\.',
        r'Oh right.*?\.',
        r'Maybe.*?\.',
        r'I should check.*?\.',
        r'Alternatively.*?\.',
        r'But given that.*?\.',
        r'In conclusion.*?\.',
        r'To fine-tune.*?\.',
        r'Note that.*?\.',
        r'This example.*?\.',
        r'But first.*?\.',
        r'Another thing.*?\.',
        r'Moreover.*?\.',
        r'However.*?\.',
        r'So, in the code.*?\.',
        r'But this requires.*?\.',
        r'Putting it all together.*?\.',
        r'Okay, the user mentioned.*?\.',
        r'First, I need to understand.*?\.',
        r'I need to understand.*?\.',
        r'The user wants.*?\.',
        r'Based on the user.*?\.',
        r'Looking at the user.*?\.',
        r'From what I can see.*?\.',
        r'It seems like.*?\.',
        r'It appears that.*?\.',
        r'Let me provide.*?\.',
        r'Here\'s what I can.*?\.'
    ]
    
    cleaned_content = content
    for pattern in thinking_patterns:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove repetitive clarification questions
    cleaned_content = re.sub(r'What kind of.*?\?', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL)
    cleaned_content = re.sub(r'Could you clarify.*?\?', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL)
    cleaned_content = re.sub(r'Can you specify.*?\?', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL)
    cleaned_content = re.sub(r'What exactly.*?\?', '', cleaned_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up multiple newlines and spaces
    cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
    cleaned_content = re.sub(r'^\s+', '', cleaned_content, flags=re.MULTILINE)
    
    return cleaned_content.strip()

# Serve the main page
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Favicon is handled by static files

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Validate model compatibility
        if not request.model:
            raise HTTPException(status_code=400, detail="Model must be specified")
        
        # Check if model supports chat (basic validation)
        if 'embed' in request.model.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model}' is an embedding model and cannot be used for chat. Please select a chat-compatible model."
            )
        
        # Get or create user ID
        user_id = request.user_id or get_or_create_user_id()
        
        # Check user session limits
        user_sessions = [s for s in chat_sessions.values() if s.get('user_id') == user_id]
        if len(user_sessions) >= settings.max_sessions_per_user:
            raise HTTPException(
                status_code=429, 
                detail=f"Maximum sessions per user ({settings.max_sessions_per_user}) exceeded. Please delete some old sessions."
            )
        
        # Get or create session
        if request.session_id and request.session_id in chat_sessions:
            session_id = request.session_id
            session = chat_sessions[session_id]
            
            # Check if session belongs to user
            if session.get('user_id') and session['user_id'] != user_id:
                raise HTTPException(status_code=403, detail="Access denied to this session")
        else:
            # Create new session
            session = create_new_session(request.model, user_id=user_id)
            session_id = session["session_id"]
            active_sessions[user_id] = session_id
        
        # Check message limits
        if session["message_count"] >= settings.max_messages_per_session:
            raise HTTPException(
                status_code=429,
                detail=f"Maximum messages per session ({settings.max_messages_per_session}) exceeded. Please start a new session."
            )
        
        # Add user message to session
        user_message_id = add_message_to_session(session_id, "user", request.message, request.model)
        
        # Update session title if it's the first message
        update_session_title(session_id, request.message)
        
        logger.info("Processing chat request", 
                   session_id=session_id, 
                   model=request.model, 
                   message_length=len(request.message),
                   user_id=user_id)
        
        # Prepare conversation context for Ollama
        messages = []
        
        # Add system message for better response quality (always include for consistent behavior)
        messages.append({
            'role': 'system',
            'content': 'You are a professional coding assistant. Provide direct, actionable solutions without thinking out loud. When asked for code: 1) Give the complete, working code immediately 2) Use proper markdown formatting with ```language blocks 3) Add brief, clear explanations 4) Do not ask for clarification unless absolutely necessary 5) Never use phrases like "Let me think", "I need to understand", "First, I remember", or similar internal monologue language. Be concise and professional.'
        })
        
        for msg in session["messages"][-10:]:  # Last 10 messages for context
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        # Prepare options
        options = {}
        if request.temperature is not None:
            options['temperature'] = request.temperature
        if request.max_tokens is not None:
            options['num_predict'] = request.max_tokens
        
        # Generate response with retry logic and better error handling
        max_retries = settings.ollama_max_retries
        last_error = None
        
        for attempt in range(max_retries):
            try:
                OLLAMA_REQUESTS.inc()
                response = ollama.chat(
                    model=request.model, 
                    messages=messages,
                    options=options
                )
                break
            except Exception as e:
                last_error = e
                OLLAMA_ERRORS.inc()
                error_msg = str(e).lower()
                
                logger.warning("Ollama request failed", 
                             attempt=attempt + 1, 
                             max_retries=max_retries, 
                             error=str(e),
                             model=request.model)
                
                # Check for specific error types
                if 'does not support chat' in error_msg or 'embed' in error_msg:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{request.model}' is not compatible with chat. Please select a different model."
                    )
                elif 'not found' in error_msg:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model '{request.model}' not found. Please check if it's properly installed."
                    )
                elif attempt == max_retries - 1:
                    # Last attempt failed
                    logger.error("All Ollama retry attempts failed", 
                               model=request.model, 
                               error=str(e))
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to generate response after {max_retries} attempts. Error: {str(e)}"
                    )
                else:
                    # Wait before retry with exponential backoff
                    wait_time = min(2 ** attempt, 10)  # Max 10 seconds
                    await asyncio.sleep(wait_time)
        
        # Add assistant response to session
        assistant_content = response['message']['content']
        # Clean the response to remove thinking patterns
        cleaned_content = clean_ai_response(assistant_content)
        assistant_message_id = add_message_to_session(session_id, "assistant", cleaned_content, request.model)
        
        return ChatResponse(
            response=cleaned_content,
            model=request.model,
            session_id=session_id,
            message_id=assistant_message_id,
            usage=response.get('eval_count', 0),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/sessions/new")
async def create_new_session_endpoint(request: NewSessionRequest):
    """Create a new chat session"""
    try:
        session = create_new_session(request.model, request.title)
        return {
            "success": True,
            "session": session
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation error: {str(e)}")

@app.get("/api/sessions")
async def get_sessions():
    """Get all chat sessions"""
    try:
        sessions = []
        for session_id, session in chat_sessions.items():
            sessions.append(SessionInfo(
                session_id=session_id,
                title=session["title"],
                model=session["model"],
                created_at=session["created_at"],
                updated_at=session["updated_at"],
                message_count=session["message_count"]
            ))
        
        # Sort by most recent
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sessions retrieval error: {str(e)}")

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific chat session with messages"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = chat_sessions[session_id]
        return {
            "session": session,
            "messages": session["messages"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session retrieval error: {str(e)}")

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        del chat_sessions[session_id]
        
        # Remove from active sessions if present
        for user_id, active_session_id in list(active_sessions.items()):
            if active_session_id == session_id:
                del active_sessions[user_id]
        
        return {"success": True, "message": "Session deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session deletion error: {str(e)}")

@app.put("/api/sessions/{session_id}/title")
async def update_session_title_endpoint(session_id: str, title_update: dict):
    """Update session title"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        new_title = title_update.get("title", "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        
        chat_sessions[session_id]["title"] = new_title
        chat_sessions[session_id]["updated_at"] = datetime.now().isoformat()
        
        return {"success": True, "title": new_title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title update error: {str(e)}")

@app.get("/api/models")
async def get_models():
    """Get available Ollama models with compatibility filtering"""
    try:
        models_response = ollama.list()
        models = []
        
        # Define known embedding models that should be filtered out
        embedding_models = {
            'nomic-embed-text', 'nomic-embed', 'all-minilm', 'all-mpnet-base-v2',
            'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'
        }
        
        # Handle both object and dictionary responses
        if hasattr(models_response, 'models') and models_response.models:
            for model in models_response.models:
                model_name = getattr(model, 'model', '')
                # Filter out embedding models and models with 'embed' in name
                if (model_name and 
                    not any(embed_name in model_name.lower() for embed_name in embedding_models) and
                    'embed' not in model_name.lower()):
                    models.append({
                        "name": model_name,
                        "size": getattr(model, 'size', 0),
                        "modified_at": str(getattr(model, 'modified_at', '')),
                        "digest": getattr(model, 'digest', '')[:12] if hasattr(model, 'digest') else '',
                        "type": "chat"  # Mark as chat-compatible
                    })
        elif isinstance(models_response, dict) and 'models' in models_response:
            for model in models_response['models']:
                model_name = model.get('model', '')
                # Filter out embedding models and models with 'embed' in name
                if (model_name and 
                    not any(embed_name in model_name.lower() for embed_name in embedding_models) and
                    'embed' not in model_name.lower()):
                    models.append({
                        "name": model_name,
                        "size": model.get('size', 0),
                        "modified_at": model.get('modified_at', ''),
                        "digest": model.get('digest', '')[:12] if model.get('digest') else '',
                        "type": "chat"  # Mark as chat-compatible
                    })
        
        # Sort models by name for better UX
        models.sort(key=lambda x: x['name'].lower())
        
        return {"models": models}
        
    except Exception as e:
        print(f"Models error: {str(e)}")
        return {"models": [], "error": str(e)}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        ollama.list()
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy", 
            "ollama": "connected",
            "sessions_count": len(chat_sessions),
            "active_sessions_count": len(active_sessions),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy", 
            "ollama": "disconnected", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# RAG Endpoints
@app.post("/api/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for RAG"""
    try:
        # Save uploaded file
        file_info = await file_service.save_uploaded_file(file)
        
        # Process document with RAG service
        result = await rag_service.process_document(
            file_info["file_path"], 
            file_info["original_name"]
        )
        
        if result["success"]:
            return DocumentUploadResponse(
                success=True,
                document_id=result["document_id"],
                file_name=result["file_name"],
                chunk_count=result["chunk_count"],
                total_length=result["total_length"],
                message="Document uploaded and processed successfully"
            )
        else:
            # Clean up file if processing failed
            file_service.delete_file(file_info["file_id"], file_info["file_type"])
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error uploading document", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query documents using RAG"""
    try:
        # Get relevant document context
        context = await rag_service.get_document_context(
            request.query, 
            max_chunks=request.max_chunks
        )
        
        if not context:
            raise HTTPException(
                status_code=400, 
                detail="No documents available for RAG queries. Please upload some documents first."
            )
        
        # Prepare conversation with context
        messages = [
            {
                'role': 'system',
                'content': f"""You are a helpful assistant that answers questions based on the provided document context. 
                Use only the information from the context to answer questions. If the context doesn't contain enough 
                information to answer the question, say so. Here is the relevant context:

                {context}

                Answer the user's question based on this context."""
            },
            {
                'role': 'user',
                'content': request.query
            }
        ]
        
        # Generate response using Ollama
        options = {'temperature': request.temperature}
        
        response = ollama.chat(
            model=request.model,
            messages=messages,
            options=options
        )
        
        # Get search results for context tracking
        search_results = await rag_service.search_documents(request.query, top_k=request.max_chunks)
        
        return RAGQueryResponse(
            response=response['message']['content'],
            query=request.query,
            model=request.model,
            context_used=search_results,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in RAG query", error=str(e))
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.get("/api/rag/documents")
async def get_documents():
    """Get list of all processed documents"""
    try:
        documents = rag_service.get_document_list()
        return {"documents": documents}
    except Exception as e:
        logger.error("Error getting documents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@app.delete("/api/rag/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its embeddings"""
    try:
        # Get document info for file cleanup
        documents = rag_service.get_document_list()
        doc_info = next((doc for doc in documents if doc["id"] == document_id), None)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from RAG service
        success = rag_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
        # Clean up file storage
        file_service.cleanup_orphaned_files([doc["id"] for doc in documents if doc["id"] != document_id])
        
        return {"success": True, "message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting document", error=str(e))
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.get("/api/rag/stats")
async def get_rag_stats():
    """Get RAG service statistics"""
    try:
        rag_stats = rag_service.get_document_stats()
        file_stats = file_service.get_storage_stats()
        
        return {
            "rag_stats": rag_stats,
            "file_stats": file_stats
        }
    except Exception as e:
        logger.error("Error getting RAG stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/api/rag/search")
async def search_documents(query: str, top_k: int = 5):
    """Search documents using semantic similarity"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = await rag_service.search_documents(query, top_k=top_k)
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error searching documents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/test-markdown")
async def test_markdown():
    """Test endpoint to verify markdown rendering"""
    test_content = """## Delhi 2‑Night, 3‑Day Itinerary

Below is a **ready‑to‑go 2‑night Delhi itinerary** that balances the city's must‑see heritage sites, a taste of local cuisine, a little shopping, and a touch of modern culture. Feel free to tweak the timings or swap a few spots to match your interests or travel style.

---

## 1️⃣ Pre‑Trip Checklist

| Item | Why It Matters | Quick Tips |
|------|----------------|------------|
| **Passport & Visa** | Needed for all international arrivals | Keep a photocopy in your luggage and a digital copy in your phone. |
| **Currency** | INR is the local currency | ATM cash is available everywhere. Card acceptance is widespread. |
| **SIM Card** | Stay connected for maps, booking, and local rides | Buy a 5‑day data SIM at the airport (Aircel, Jio, or Airtel). |
| **Weather** | Delhi can be hot (summer) or chilly (winter) | Pack a light jacket or sunglasses & sunscreen. |
| **Health** | Hydration & food safety | Carry bottled water and basic meds (pain reliever, motion sickness). |
| **Travel Insurance** | Covers medical & accidental events | Check that it includes COVID‑19 coverage if needed. |

---

## 2️⃣ Arrival & Accommodation

| Option | Approx. Cost (₹/night) | Highlights | Best For |
|--------|------------------------|------------|----------|
| **Luxury** – *The Imperial, New Delhi* | ₹15,000–₹20,000 | Classic colonial décor, 5‑star amenities, central location (Connaught Place). | Honeymooners, business travelers |
| **Mid‑Range** – *JW Marriott Hotel, New Delhi* | ₹8,000–₹12,000 | Rooftop pool, great spa, easy metro access. | Couples, solo travelers |
| **Budget** – *The Lodhi, New Delhi* | ₹3,000–₹5,000 | Boutique vibe, close to Old Delhi, free Wi‑Fi. | Backpackers, value‑seekers |
| **Airbnb** – Private room in Connaught Place | ₹2,000–₹4,000 | Local vibe, often includes breakfast. | Solo travelers, groups |

**Booking Tip** – Book via the hotel's own site or a reputable OTA (MakeMyTrip, Booking.com). Look for "free cancellation" if your schedule might change.

**Transport from the Airport**
- **Metro**: Airport Express Line (₹60) to New Delhi Station → transfer to Yellow Line to your hotel.
- **Airport Taxi**: ₹400–₹600 (metered).
- **Ride‑hailing**: Uber/Lyft – ₹350–₹500.
- **Airport Shuttle**: Many hotels offer free or paid shuttles; check in advance.

---

## 3️⃣ 2‑Night Itinerary (Sample)

> **Assumptions** – Arrival on Day 1 afternoon, departure on Day 3 morning.
> **Adjust** – If you arrive early or leave late, add a quick morning/afternoon visit.

### Day 1 – Arrival & Evening Exploration

| Time | Activity | Notes |
|------|----------|-------|
| **12:30 pm** | Check‑in & freshen up | Drop luggage at concierge. |
| **1:30 pm** | Lunch at *Karim's* (Old Delhi) | Try *mutton seekh kebab* & *nawabi haleem*. |
| **3:00 pm** | Walk to *Jama Masjid* & *Red Fort* | 30‑min walk; explore the bazaar on the way. |
| **5:00 pm** | Ride to *India Gate* (via Metro – Yellow Line) | Sunset photo spot. |
| **6:30 pm** | Dinner at *Bukhara* (ITC Maurya) | Reservations recommended; famous for *tandoori* dishes. |
| **8:30 pm** | Night walk at *Connaught Place* (shopping & cafés) | Light stroll, grab a chai at *Café Zauq*. |

**Optional** – If you're staying in Connaught Place, consider a *Rickshaw tour* of Old Delhi (₹200–₹300).

---

### Day 2 – Full Day of Culture & Modernity

| Time | Activity | Notes |
|------|----------|-------|
| **7:30 am** | Breakfast at hotel (or *The Indian Coffee House* for a local vibe). | |
| **9:00 am** | Visit *Humayun's Tomb* (Metro – Yellow Line → *JNU*). | UNESCO heritage site. |
| **10:30 am** | Walk to *Lotus Temple* (₹100 entry). | Meditation space; no photos inside. |
| **12:00 pm** | Lunch at *Sukhi* (Punjabi restaurant) | Try *tandoori chicken* & *makki di roti*. |
| **1:30 pm** | Explore *Qutub Minar* (Metro – Yellow Line → *Qutub Minar*). | 12‑m‑tall minaret; climb the 5th floor for a view. |
| **3:00 pm** | Coffee break at *The Park Café* (Park Street, New Delhi). | |
| **4:00 pm** | Visit *Akshardham Temple* (Metro – Violet Line → *Akshardham*). | Free entry; watch the evening light & sound show (8:30 pm). |
| **6:30 pm** | Dinner at *Bengal Street* (street‑food stalls). | Try *kathi rolls*, *samosas*, *jalebi*. |
| **8:00 pm** | Light evening stroll at *Raj Ghat* (Memorial to Mahatma Gandhi). | Quiet, reflective. |
| **9:00 pm** | Return to hotel (Metro or cab). | |

**Pro‑Tip** – Use the **Delhi Metro** app or Google Maps for real‑time updates; the network is reliable and covers most attractions.

---

### Day 3 – Departure

| Time | Activity | Notes |
|------|----------|-------|
| **6:30 am** | Breakfast at hotel / local café | |
| **7:30 am** | Check‑out & collect luggage | |
| **8:00 am** | Head to the airport | Use metro or pre‑book a taxi. |
| **10:00 am** | Flight departure | |

---

## 4️⃣ Dining Highlights (Beyond the Itinerary)

| Dish | Where | Why |
|------|-------|-----|
| **Biryani** | *Baba Nandlal* (Old Delhi) | Classic Mughlai rice dish. |
| **Chaat** | *Chaat Street* (Chandni Chowk) | Mix of tangy, sweet, and spicy. |
| **Thali** | *Saravana Bhavan* (Connaught Place) | South‑Indian vegetarian feast. |
| **Street‑Food** | *Khan Market* (Pav Bhaji, Momos) | Trendy, quick bites. |

---

## 5️⃣ Budget Snapshot (₹ ≈ US$)

| Category | Approx. Cost (₹) | Notes |
|----------|------------------|-------|
| **Accommodation (2 nights)** | 3,000–20,000 | Depends on class. |
| **Meals** | 1,500–3,000 | 2–3 meals per day. |
| **Transport (metro + taxis)** | 1,000–2,000 | |
| **Entry Fees** | 500–1,000 | Red Fort, Qutub Minar, Akshardham. |
| **Misc. (shopping, tips)** | 1,000–2,000 | |
| **Total** | 7,000–28,000 | Roughly US$95–380 |

---

## 6️⃣ Tips & Tricks

| Topic | Advice |
|-------|--------|
| **Safety** | Keep valuables in a money belt. Avoid empty lanes at night. |
| **Language** | English is widely spoken in hotels & metro. Learn a few Hindi phrases (e.g., *Namaste*, *Shukriya*). |
| **Water** | Drink bottled water only. |
| **Dress** | Modest clothing (especially near religious sites). |
| **Connectivity** | Free Wi‑Fi in most hotels, but a local SIM gives you reliable data. |
| **Cash vs. Card** | Many places accept cards; carry some cash for street vendors and small shops. |

---

## 7️⃣ Quick FAQ

| Question | Answer |
|----------|--------|
| **Is Delhi safe for solo travelers?** | Yes, but stay alert, especially at night. |
| **What's the best way to get around?** | Delhi Metro is the most efficient; use ride‑hailing apps for short distances. |
| **Can I visit all sites in 2 days?** | Yes, but you'll need to be early and efficient. |
| **Do I need a visa?** | Yes, unless you're from a visa‑exempt country. Apply online (e‑visa) before departure. |

---

### Final Thought

Delhi is a city of contrasts: ancient monuments next to buzzing markets, quiet temples beside neon‑lit malls. With this 2‑night plan you'll hit the highlights, taste the flavors, and feel the pulse of India's capital. Have a fantastic trip! 🚀

*Happy travels!*"""
    
    return {"response": test_content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)