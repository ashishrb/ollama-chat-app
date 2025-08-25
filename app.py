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

# Import settings with better error handling
try:
    from config import settings
    print(f"✅ Config loaded - Environment: {getattr(settings, 'debug', 'unknown')}")
except ImportError as e:
    print(f"⚠️ Config import failed: {e}")
    # Fallback settings if config.py doesn't exist
    class Settings:
        debug = os.getenv("DEBUG", "true").lower() == "true"  # Default to True for development
        app_name = "Ollama Chat App with RAG"
        app_version = "1.0.0"
        # More permissive CORS for development
        cors_origins = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
        max_sessions_per_user = int(os.getenv("MAX_SESSIONS_PER_USER", "100"))
        max_messages_per_session = int(os.getenv("MAX_MESSAGES_PER_SESSION", "1000"))
        session_ttl = int(os.getenv("SESSION_TTL", str(86400 * 7)))  # 7 days
        session_cleanup_interval = int(os.getenv("SESSION_CLEANUP_INTERVAL", "3600"))  # 1 hour
        # More lenient rate limiting for development
        rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "300"))  # Increased from 100
        rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # 1 minute
        ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "30"))
        ollama_max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
    
    settings = Settings()
    print(f"✅ Fallback config loaded - Debug: {settings.debug}")

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
rate_limit_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=settings.rate_limit_requests))

# Enhanced rate limiting with better error handling
def check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit - more lenient implementation"""
    try:
        now = time.time()
        client_requests = rate_limit_store[client_id]
        
        # Remove old requests outside the window
        while client_requests and now - client_requests[0] > settings.rate_limit_window:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= settings.rate_limit_requests:
            logger.warning(f"Rate limit exceeded for client {client_id}", 
                         requests=len(client_requests), 
                         limit=settings.rate_limit_requests)
            return False
        
        # Add current request
        client_requests.append(now)
        return True
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        return True  # Allow request if rate limiting fails

def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting with better error handling"""
    try:
        # Use X-Forwarded-For for proxy support, fallback to client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Safely get client host
        if request.client and hasattr(request.client, 'host') and request.client.host:
            return request.client.host
        
        # Fallback to request headers
        host = request.headers.get("Host", "localhost")
        return host.split(":")[0] if ":" in host else host
        
    except Exception as e:
        logger.warning(f"Failed to get client ID: {e}")
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

# Pydantic models with enhanced validation
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
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if not v or not v.strip():
            raise ValueError('Model must be specified')
        # Remove the embedding model restriction that was causing issues
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
        if not v or not v.strip():
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
    logger.info("Starting Ollama Chat App", 
                version=settings.app_version,
                debug_mode=settings.debug)
    
    # Initialize RAG and File services
    global rag_service, file_service
    rag_service = RAGService()
    file_service = FileService()
    
    try:
        # Test Ollama connection
        models = ollama.list()
        logger.info("Ollama connection successful", models_count=len(models.get('models', [])))
        
        # Initialize RAG service
        await rag_service.initialize()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error("Ollama connection failed", error=str(e))
        logger.info("App will continue without Ollama - some features may not work")
    
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
    description="Advanced chat application with RAG capabilities and document management",
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug
)

# Add production middleware with better host configuration
app.add_middleware(GZipMiddleware, minimum_size=1000)

# IMPROVED: More flexible TrustedHostMiddleware configuration
environment = os.getenv("ENVIRONMENT", "development").lower()
if not settings.debug and environment == "production":
    # Only add strict host restrictions in production
    allowed_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
    
    # Add custom hosts from environment
    custom_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")
    allowed_hosts.extend([host.strip() for host in custom_hosts if host.strip()])
    
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=allowed_hosts
    )
    logger.info("TrustedHostMiddleware enabled", allowed_hosts=allowed_hosts)
else:
    logger.info("TrustedHostMiddleware disabled (development mode)")

# Serve static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    logger.warning("Static directory not found")

# IMPROVED: More flexible CORS configuration
cors_origins = settings.cors_origins
if isinstance(cors_origins, str):
    cors_origins = [cors_origins]

logger.info("CORS origins configured", origins=cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Enhanced middleware for request timing and metrics
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    try:
        # Enhanced rate limiting with better error handling
        should_rate_limit = (
            not request.url.path.startswith("/static") and 
            request.url.path not in ["/", "/favicon.ico", "/health"] and
            request.url.path.startswith("/api/") and
            request.method != "OPTIONS"  # Skip OPTIONS requests
        )
        
        if should_rate_limit:
            try:
                client_id = get_client_id(request)
                if not check_rate_limit(client_id):
                    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=429).inc()
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded. Please try again later.",
                            "retry_after": settings.rate_limit_window
                        }
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
        response.headers["X-App-Version"] = settings.app_version
        
        return response
        
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}")
        # Return a proper error response instead of letting it crash
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

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
    try:
        return FileResponse("static/index.html")
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"detail": "Frontend not found. Please ensure static/index.html exists."}
        )

# Health check endpoint (moved before other endpoints for faster access)
@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Test Ollama connection
        ollama_status = "connected"
        ollama_models = 0
        try:
            models = ollama.list()
            ollama_models = len(models.get('models', []))
        except Exception as e:
            ollama_status = f"disconnected: {str(e)}"
        
        # Get system metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
        except Exception as e:
            cpu_percent = 0
            memory = type('obj', (object,), {'percent': 0})
            disk = type('obj', (object,), {'percent': 0})
        
        return {
            "status": "healthy", 
            "ollama": ollama_status,
            "ollama_models": ollama_models,
            "sessions_count": len(chat_sessions),
            "active_sessions_count": len(active_sessions),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "app_version": settings.app_version,
            "debug_mode": settings.debug,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info("Processing chat request", 
                   model=request.model, 
                   message_length=len(request.message))
        
        # Enhanced model validation
        if not request.model or not request.model.strip():
            raise HTTPException(status_code=400, detail="Model must be specified")
        
        # More flexible model compatibility check
        model_lower = request.model.lower()
        if 'embed' in model_lower and not any(chat_word in model_lower for chat_word in ['chat', 'instruct', 'code']):
            logger.warning(f"Embedding model used for chat: {request.model}")
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model}' appears to be an embedding model. For chat, please select a chat-compatible model like 'llama3.2:3b', 'mistral:7b', or 'codellama:7b'."
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
            'content': 'You are a professional AI assistant. Provide direct, actionable solutions without thinking out loud. When asked for code: 1) Give the complete, working code immediately 2) Use proper markdown formatting with ```language blocks 3) Add brief, clear explanations 4) Do not ask for clarification unless absolutely necessary 5) Never use phrases like "Let me think", "I need to understand", "First, I remember", or similar internal monologue language. Be concise and professional.'
        })
        
        # Add conversation history (last 10 messages for context)
        for msg in session["messages"][-10:]:
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
        
        # Generate response with enhanced retry logic
        max_retries = settings.ollama_max_retries
        last_error = None
        
        for attempt in range(max_retries):
            try:
                OLLAMA_REQUESTS.inc()
                
                # Test if model exists first
                try:
                    available_models = ollama.list()
                    model_names = [m.get('model', '') for m in available_models.get('models', [])]
                    if request.model not in model_names:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Model '{request.model}' not found. Available models: {', '.join(model_names[:5])}"
                        )
                except Exception as model_check_error:
                    logger.warning(f"Could not verify model existence: {model_check_error}")
                
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
                if 'does not support chat' in error_msg or 'embedding' in error_msg:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{request.model}' does not support chat functionality. Please select a different model."
                    )
                elif 'not found' in error_msg or 'no such file' in error_msg:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model '{request.model}' not found. Please check if it's properly installed with: ollama pull {request.model}"
                    )
                elif 'connection' in error_msg or 'refused' in error_msg:
                    raise HTTPException(
                        status_code=503,
                        detail="Ollama service is not available. Please ensure Ollama is running."
                    )
                elif attempt == max_retries - 1:
                    # Last attempt failed
                    logger.error("All Ollama retry attempts failed", 
                               model=request.model, 
                               error=str(e))
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to generate response after {max_retries} attempts. Please try again or check if the model is working properly."
                    )
                else:
                    # Wait before retry with exponential backoff
                    wait_time = min(2 ** attempt, 10)  # Max 10 seconds
                    await asyncio.sleep(wait_time)
        
        # Process response
        if not response or 'message' not in response or 'content' not in response.get('message', {}):
            raise HTTPException(
                status_code=500,
                detail="Invalid response from Ollama. Please try again."
            )
        
        # Add assistant response to session
        assistant_content = response['message']['content']
        # Clean the response to remove thinking patterns
        cleaned_content = clean_ai_response(assistant_content)
        assistant_message_id = add_message_to_session(session_id, "assistant", cleaned_content, request.model)
        
        logger.info("Chat response generated successfully", 
                   session_id=session_id,
                   response_length=len(cleaned_content))
        
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
        logger.error("Unexpected chat error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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
        logger.error("Session creation error", error=str(e))
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
        logger.error("Sessions retrieval error", error=str(e))
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session retrieval error", error=str(e))
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
        
        # Remove from creation times
        if session_id in session_creation_times:
            del session_creation_times[session_id]
        
        # Update metrics
        TOTAL_SESSIONS.set(len(chat_sessions))
        ACTIVE_SESSIONS.set(len(active_sessions))
        
        logger.info("Session deleted", session_id=session_id)
        return {"success": True, "message": "Session deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session deletion error", error=str(e))
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Title update error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Title update error: {str(e)}")

@app.get("/api/models")
async def get_models():
    """Get available Ollama models with enhanced compatibility filtering"""
    try:
        models_response = ollama.list()
        models = []
        
        # Define known embedding models that should be filtered out for chat
        embedding_indicators = {
            'nomic-embed-text', 'nomic-embed', 'all-minilm', 'all-mpnet-base-v2',
            'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large',
            'sentence-transformers', 'bge-', 'gte-'
        }
        
        # Handle both object and dictionary responses
        raw_models = []
        if hasattr(models_response, 'models') and models_response.models:
            raw_models = models_response.models
        elif isinstance(models_response, dict) and 'models' in models_response:
            raw_models = models_response['models']
        
        for model in raw_models:
            if hasattr(model, 'model'):
                model_name = model.model
                model_size = getattr(model, 'size', 0)
                model_modified = str(getattr(model, 'modified_at', ''))
                model_digest = getattr(model, 'digest', '')[:12] if hasattr(model, 'digest') else ''
            else:
                model_name = model.get('model', '')
                model_size = model.get('size', 0)
                model_modified = model.get('modified_at', '')
                model_digest = model.get('digest', '')[:12] if model.get('digest') else ''
            
            if not model_name:
                continue
            
            # Enhanced filtering logic
            model_lower = model_name.lower()
            is_embedding = any(embed_name in model_lower for embed_name in embedding_indicators)
            is_chat_compatible = any(chat_word in model_lower for chat_word in ['chat', 'instruct', 'code'])
            
            # Include model if it's not an embedding model OR if it has chat indicators
            if not is_embedding or is_chat_compatible:
                models.append({
                    "name": model_name,
                    "size": model_size,
                    "modified_at": model_modified,
                    "digest": model_digest,
                    "type": "chat",
                    "is_embedding": is_embedding and not is_chat_compatible
                })
        
        # Sort models by name for better UX
        models.sort(key=lambda x: x['name'].lower())
        
        logger.info(f"Found {len(models)} available models")
        return {"models": models}
        
    except Exception as e:
        logger.error("Models retrieval error", error=str(e))
        return {
            "models": [], 
            "error": str(e),
            "message": "Could not retrieve models. Please ensure Ollama is running and accessible."
        }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    try:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error("Metrics generation error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate metrics")

# RAG Endpoints with enhanced error handling
@app.post("/api/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for RAG"""
    try:
        logger.info("Processing document upload", filename=file.filename, content_type=file.content_type)
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file
        file_info = await file_service.save_uploaded_file(file)
        
        # Process document with RAG service
        result = await rag_service.process_document(
            file_info["file_path"], 
            file_info["original_name"]
        )
        
        if result["success"]:
            logger.info("Document processed successfully", 
                       document_id=result["document_id"],
                       chunks=result["chunk_count"])
            
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
        logger.error("Error uploading document", error=str(e), filename=getattr(file, 'filename', 'unknown'))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query documents using RAG"""
    try:
        logger.info("Processing RAG query", query_length=len(request.query), model=request.model)
        
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
                information to answer the question, say so clearly. Be direct and professional.

                Context from documents:
                {context}

                Answer the user's question based on this context."""
            },
            {
                'role': 'user',
                'content': request.query
            }
        ]
        
        # Generate response using Ollama with retry logic
        options = {'temperature': request.temperature}
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = ollama.chat(
                    model=request.model,
                    messages=messages,
                    options=options
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to generate RAG response: {str(e)}"
                    )
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Get search results for context tracking
        search_results = await rag_service.search_documents(request.query, top_k=request.max_chunks)
        
        logger.info("RAG query processed successfully", 
                   response_length=len(response['message']['content']),
                   context_chunks=len(search_results))
        
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
        remaining_doc_ids = [doc["id"] for doc in documents if doc["id"] != document_id]
        file_service.cleanup_orphaned_files(remaining_doc_ids)
        
        logger.info("Document deleted successfully", document_id=document_id)
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
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = await rag_service.search_documents(query, top_k=top_k)
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error searching documents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Test endpoints for debugging
@app.get("/api/test-markdown")
async def test_markdown():
    """Test endpoint to verify markdown rendering"""
    test_content = """## Test Response

This is a **test response** with:

- **Bold text**
- *Italic text*
- `Code snippets`

```python
def hello_world():
    print("Hello, World!")
    return "Success"
```

### Code Block Test

```javascript
const testFunction = () => {
    console.log("Testing markdown rendering");
    return { status: "working" };
};
```

The system is working correctly!"""
    
    return {"response": test_content}

@app.get("/api/debug/config")
async def debug_config():
    """Debug endpoint to check configuration (remove in production)"""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Debug endpoint not available in production")
    
    return {
        "debug": settings.debug,
        "cors_origins": settings.cors_origins,
        "rate_limit_requests": settings.rate_limit_requests,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "allowed_hosts_env": os.getenv("ALLOWED_HOSTS", "not_set"),
        "session_count": len(chat_sessions),
        "app_version": settings.app_version
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Endpoint not found: {request.url.path}"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error("Internal server error", 
                path=request.url.path,
                method=request.method,
                error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for local development
    config = {
        "app": "app:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": settings.debug,
        "log_level": "info" if not settings.debug else "debug",
    }
    
    logger.info("Starting Ollama Chat App", **config)
    uvicorn.run(**config)