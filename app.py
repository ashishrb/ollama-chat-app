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
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
import aiofiles
from functools import lru_cache
import hashlib

# Import RAG services
from rag_service import RAGService
from file_service import FileService

from professional_prompts import (
    SystemPrompts, 
    SmartPrompts, 
    get_prompt_for_request,
    PromptConfig
)

# Import settings with better error handling
try:
    from config import settings
    print(f"✅ Config loaded - Environment: {getattr(settings, 'debug', 'unknown')}")
except ImportError as e:
    print(f"⚠️ Config import failed: {e}")
    # Fallback settings optimized for performance
    class Settings:
        debug = os.getenv("DEBUG", "true").lower() == "true"
        app_name = "Ollama Chat App with RAG"
        app_version = "1.0.0"
        cors_origins = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
        max_sessions_per_user = int(os.getenv("MAX_SESSIONS_PER_USER", "100"))
        max_messages_per_session = int(os.getenv("MAX_MESSAGES_PER_SESSION", "1000"))
        session_ttl = int(os.getenv("SESSION_TTL", str(86400 * 7)))
        session_cleanup_interval = int(os.getenv("SESSION_CLEANUP_INTERVAL", "3600"))
        # Performance optimized rate limiting
        rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "300"))
        rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "30"))
        ollama_max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        
        # Performance settings
        enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
        max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))  # Reduced for speed
        chunk_batch_size = int(os.getenv("CHUNK_BATCH_SIZE", "5"))  # Parallel processing

        
    
    settings = Settings()
    print(f"✅ Performance-optimized fallback config loaded - Debug: {settings.debug}")

# Configure structured logging with performance optimizations
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

# Performance metrics
# Clear any existing metrics to avoid collisions
try:
    REGISTRY._collector_to_names.clear()
    REGISTRY._names_to_collectors.clear()
except:
    pass

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
RESPONSE_GENERATION_TIME = Histogram('response_generation_seconds', 'Time to generate AI response')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
ACTIVE_SESSIONS = Gauge('active_sessions', 'Number of active sessions')
TOTAL_SESSIONS = Gauge('total_sessions', 'Total number of sessions')
OLLAMA_REQUESTS = Counter('ollama_requests_total', 'Total Ollama API requests')
OLLAMA_ERRORS = Counter('ollama_errors_total', 'Total Ollama API errors')

# Global storage with TTL support and caching
chat_sessions: Dict[str, Dict[str, Any]] = {}
active_sessions: Dict[str, str] = {}
session_creation_times: Dict[str, float] = {}
rate_limit_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=settings.rate_limit_requests))

# Performance caches
response_cache: Dict[str, Dict[str, Any]] = {}  # Simple in-memory cache
model_warmup_cache: Dict[str, bool] = {}  # Track warmed-up models

# OPTIMIZED: Response caching
def get_cache_key(message: str, model: str, temperature: float, context: str = "") -> str:
    """Generate cache key for responses"""
    content = f"{message}:{model}:{temperature}:{context[:200]}"  # Limit context for key
    return hashlib.md5(content.encode()).hexdigest()

def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached response if available and not expired"""
    if not settings.enable_caching:
        return None
        
    if cache_key in response_cache:
        cached = response_cache[cache_key]
        if time.time() - cached['timestamp'] < settings.cache_ttl:
            CACHE_HITS.labels(cache_type='response').inc()
            return cached['data']
        else:
            # Expired, remove from cache
            del response_cache[cache_key]
    
    CACHE_MISSES.labels(cache_type='response').inc()
    return None

def cache_response(cache_key: str, response_data: Dict[str, Any]):
    """Cache response data"""
    if not settings.enable_caching:
        return
        
    response_cache[cache_key] = {
        'data': response_data,
        'timestamp': time.time()
    }
    
    # Simple cache size management
    if len(response_cache) > 1000:  # Max 1000 cached responses
        # Remove oldest 100 entries
        oldest_keys = sorted(response_cache.keys(), 
                           key=lambda k: response_cache[k]['timestamp'])[:100]
        for key in oldest_keys:
            del response_cache[key]

# OPTIMIZED: Rate limiting with better performance
def check_rate_limit(client_id: str) -> bool:
    """Optimized rate limiting check"""
    try:
        now = time.time()
        client_requests = rate_limit_store[client_id]
        
        # Batch cleanup for better performance
        cutoff_time = now - settings.rate_limit_window
        while client_requests and client_requests[0] < cutoff_time:
            client_requests.popleft()
        
        if len(client_requests) >= settings.rate_limit_requests:
            return False
        
        client_requests.append(now)
        return True
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        return True  # Allow request if rate limiting fails

def get_client_id(request: Request) -> str:
    """Optimized client ID extraction"""
    try:
        # Quick header check
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        if request.client and request.client.host:
            return request.client.host
        
        return request.headers.get("host", "unknown").split(":")[0]
    except Exception:
        return "unknown"

# OPTIMIZED: Session cleanup
async def cleanup_expired_sessions():
    """Optimized session cleanup"""
    now = time.time()
    expired_sessions = [
        session_id for session_id, creation_time in session_creation_times.items()
        if now - creation_time > settings.session_ttl
    ]
    
    if not expired_sessions:
        return
    
    # Batch deletion
    for session_id in expired_sessions:
        chat_sessions.pop(session_id, None)
        session_creation_times.pop(session_id, None)
        
        # Remove from active sessions
        for user_id, active_session_id in list(active_sessions.items()):
            if active_session_id == session_id:
                del active_sessions[user_id]
                break
    
    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    TOTAL_SESSIONS.set(len(chat_sessions))
    ACTIVE_SESSIONS.set(len(active_sessions))

# OPTIMIZED: Pydantic models with performance tweaks
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)  # Reduced max length
    model: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)  # Reduced for speed
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()[:8000]  # Truncate if too long

class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: str
    message_id: str
    usage: Optional[int] = None
    timestamp: str
    cached: bool = False  # Indicate if response was cached

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

# RAG Models (optimized)
class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=3000)  # Reduced for performance
    model: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_chunks: int = Field(3, ge=1, le=5)  # Reduced max chunks
    include_context: bool = Field(True)

class RAGQueryResponse(BaseModel):
    response: str
    query: str
    model: str
    context_used: List[Dict[str, Any]]
    timestamp: str
    cached: bool = False

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
    # Startup with performance optimizations
    start_time = time.time()
    logger.info("Starting Performance-Optimized Ollama Chat App", 
                version=settings.app_version,
                debug_mode=settings.debug,
                caching_enabled=settings.enable_caching)
    
    # Initialize services
    global rag_service, file_service
    rag_service = RAGService()
    file_service = FileService()
    
    try:
        # Test Ollama connection and warm up common models
        models = ollama.list()
        model_count = len(models.get('models', []))
        logger.info("Ollama connection successful", models_count=model_count)
        
        # Warm up commonly used models (async)
        asyncio.create_task(warm_up_models())
        
        # Initialize RAG service
        await rag_service.initialize()
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error("Ollama connection failed", error=str(e))
        logger.info("App will continue without Ollama - some features may not work")
    
    # Start background tasks
    cleanup_task = asyncio.create_task(periodic_cleanup())
    cache_cleanup_task = asyncio.create_task(periodic_cache_cleanup())
    
    startup_time = time.time() - start_time
    logger.info(f"Startup completed in {startup_time:.2f}s")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ollama Chat App")
    cleanup_task.cancel()
    cache_cleanup_task.cancel()
    try:
        await asyncio.gather(cleanup_task, cache_cleanup_task, return_exceptions=True)
    except Exception:
        pass

async def warm_up_models():
    """Pre-load commonly used models for better performance"""
    # Only warmup models that are actually available
    common_models = ['qwen3:latest', 'qwen2.5-coder:1.5b', 'gpt-oss:20B']
    
    for model in common_models:
        try:
            # Simple test request to warm up model
            ollama.generate(model=model, prompt="Hello", options={'num_predict': 1})
            model_warmup_cache[model] = True
            logger.info(f"Warmed up model: {model}")
        except Exception as e:
            logger.warning(f"Failed to warm up model {model}: {e}")
        
        # Prevent overwhelming the system
        await asyncio.sleep(1)

async def periodic_cleanup():
    """Optimized periodic cleanup"""
    while True:
        try:
            await cleanup_expired_sessions()
            await asyncio.sleep(settings.session_cleanup_interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))
            await asyncio.sleep(60)

async def periodic_cache_cleanup():
    """Clean up expired cache entries"""
    while True:
        try:
            if settings.enable_caching:
                now = time.time()
                expired_keys = [
                    key for key, data in response_cache.items()
                    if now - data['timestamp'] > settings.cache_ttl
                ]
                
                for key in expired_keys:
                    del response_cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            await asyncio.sleep(3600)  # Clean every hour
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Cache cleanup error", error=str(e))
            await asyncio.sleep(3600)

app = FastAPI(
    title=settings.app_name,
    description="High-Performance chat application with RAG capabilities",
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug
)

# Middleware with performance optimizations
app.add_middleware(GZipMiddleware, minimum_size=1000)

# OPTIMIZED: Conditional TrustedHostMiddleware
environment = os.getenv("ENVIRONMENT", "development").lower()
if environment == "production":
    allowed_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
    custom_hosts = os.getenv("ALLOWED_HOSTS", "").split(",")
    allowed_hosts.extend([host.strip() for host in custom_hosts if host.strip()])
    
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    logger.info("TrustedHostMiddleware enabled", allowed_hosts=allowed_hosts)

# Serve static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# OPTIMIZED: CORS configuration
cors_origins = settings.cors_origins
if isinstance(cors_origins, str):
    cors_origins = [cors_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# OPTIMIZED: Performance middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    
    try:
        # Skip rate limiting for health checks and static files
        should_rate_limit = (
            not request.url.path.startswith("/static") and 
            request.url.path not in ["/", "/favicon.ico", "/api/health", "/metrics"] and
            request.url.path.startswith("/api/") and
            request.method != "OPTIONS"
        )
        
        if should_rate_limit:
            client_id = get_client_id(request)
            if not check_rate_limit(client_id):
                REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=429).inc()
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded", "retry_after": settings.rate_limit_window}
                )
        
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
        REQUEST_DURATION.observe(process_time)
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        response.headers["X-Cache-Enabled"] = str(settings.enable_caching)
        
        return response
        
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

def get_or_create_user_id() -> str:
    """Generate user ID"""
    return str(uuid.uuid4())

def create_new_session(model: str, title: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Optimized session creation"""
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
    
    return session

def update_session_title(session_id: str, message: str) -> None:
    """Optimized title update"""
    if session_id in chat_sessions:
        session = chat_sessions[session_id]
        if session["message_count"] == 1:
            # Quick title generation
            words = message.split()[:4]  # Reduced to 4 words for speed
            title = " ".join(words) + ("..." if len(message.split()) > 4 else "")
            session["title"] = title
            session["updated_at"] = datetime.now().isoformat()

def add_message_to_session(session_id: str, role: str, content: str, model: str) -> str:
    """Optimized message addition"""
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

# OPTIMIZED: Lightweight system prompt for faster processing
def get_optimized_system_prompt(message: str = "", model: str = "qwen3:latest", mode: str = "auto", context: str = "") -> str:
    """Get optimized system prompt based on request characteristics and environment"""
    
    # Get environment configuration
    environment = os.getenv("ENVIRONMENT", "production")
    config = PromptConfig.get_config(environment)
    
    # For RAG queries, use RAG-specific prompts
    if mode == "rag" and context:
        smart_prompts = SmartPrompts(prefer_speed=config.get("prefer_speed", True))
        return smart_prompts.get_rag_prompt(context, quality_mode=not config.get("prefer_speed", True))
    
    # For regular chat, use smart prompt selection
    if mode == "auto" or mode == "chat":
        return get_prompt_for_request(message, model, config.get("default_mode", "balanced"))
    
    # For specific modes, use system prompts
    return SystemPrompts.get_prompt(mode)

# Health check endpoint (optimized)
@app.get("/api/health")
async def health_check():
    """Ultra-fast health check"""
    try:
        # Quick Ollama check
        ollama_status = "connected" if model_warmup_cache else "connecting"
        
        return {
            "status": "healthy", 
            "ollama": ollama_status,
            "sessions_count": len(chat_sessions),
            "cache_enabled": settings.enable_caching,
            "cached_responses": len(response_cache),
            "warmed_models": len(model_warmup_cache),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}
        )

# OPTIMIZED: Context management functions
def get_optimal_context_for_model(model_name: str, messages: List[Dict], system_prompt: str) -> List[Dict]:
    """
    Enhanced context management optimized for different models, especially gpt-oss-20b
    """
    try:
        # Get model-specific context limit
        context_limit = settings.get_context_limit_for_model(model_name)
        
        # Start with system prompt
        context_messages = [{"role": "system", "content": system_prompt}]
        current_length = len(system_prompt)
        
        # If smart truncation is disabled, use simple truncation
        if not settings.enable_smart_truncation:
            return simple_context_truncation(messages, context_limit, system_prompt)
        
        # ✅ SMART TRUNCATION: Always preserve recent messages
        recent_messages = messages[-settings.preserve_recent_messages:] if len(messages) > settings.preserve_recent_messages else messages
        
        # Add recent messages first (guaranteed inclusion)
        recent_length = sum(len(msg.get('content', '')) for msg in recent_messages)
        
        if current_length + recent_length <= context_limit:
            # We can fit recent messages, now add older messages if space allows
            context_messages.extend(recent_messages)
            current_length += recent_length
            
            # Add older messages from newest to oldest
            older_messages = messages[:-settings.preserve_recent_messages] if len(messages) > settings.preserve_recent_messages else []
            
            for i in range(len(older_messages) - 1, -1, -1):
                msg = older_messages[i]
                msg_length = len(msg.get('content', ''))
                
                if current_length + msg_length + settings.context_overlap_buffer <= context_limit:
                    context_messages.insert(-len(recent_messages), msg)  # Insert before recent messages
                    current_length += msg_length
                else:
                    break
        else:
            # Recent messages don't fit, truncate them smartly
            context_messages.extend(truncate_messages_intelligently(recent_messages, context_limit - current_length))
        
        logger.info(f"Context optimized for {model_name}: {len(context_messages)} messages, {current_length} chars")
        return context_messages
        
    except Exception as e:
        logger.error(f"Error in context optimization: {e}")
        return simple_context_truncation(messages, 8000, system_prompt)

def simple_context_truncation(messages: List[Dict], limit: int, system_prompt: str) -> List[Dict]:
    """Fallback simple truncation"""
    context_messages = [{"role": "system", "content": system_prompt}]
    current_length = len(system_prompt)
    
    # Add messages from newest to oldest
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        msg_length = len(msg.get('content', ''))
        
        if current_length + msg_length <= limit:
            context_messages.insert(1, msg)  # Insert after system prompt
            current_length += msg_length
        else:
            break
    
    return context_messages

def truncate_messages_intelligently(messages: List[Dict], available_space: int) -> List[Dict]:
    """
    Intelligently truncate messages while preserving conversation flow
    """
    if not messages:
        return []
    
    # Always try to keep user-assistant pairs together
    truncated = []
    current_length = 0
    
    # Work backwards from most recent
    i = len(messages) - 1
    while i >= 0 and current_length < available_space:
        msg = messages[i]
        msg_length = len(msg.get('content', ''))
        
        # If this is a user message, try to include its assistant response too
        if msg.get('role') == 'user' and i < len(messages) - 1:
            next_msg = messages[i + 1]
            if next_msg.get('role') == 'assistant':
                pair_length = msg_length + len(next_msg.get('content', ''))
                if current_length + pair_length <= available_space:
                    truncated.insert(0, msg)
                    truncated.insert(1, next_msg)
                    current_length += pair_length
                    i -= 1  # Skip the assistant message in next iteration
                else:
                    break
            else:
                if current_length + msg_length <= available_space:
                    truncated.insert(0, msg)
                    current_length += msg_length
                else:
                    break
        else:
            if current_length + msg_length <= available_space:
                truncated.insert(0, msg)
                current_length += msg_length
            else:
                break
        
        i -= 1
    
    return truncated

def get_conversation_summary(messages: List[Dict], max_length: int = 200) -> str:
    """
    Generate a brief summary of older conversation context
    """
    if len(messages) < 4:
        return ""
    
    # Take first few and last few messages to create summary
    early_messages = messages[:2]
    
    summary_parts = []
    for msg in early_messages:
        if msg.get('role') == 'user':
            content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
            summary_parts.append(f"User asked about: {content}")
    
    summary = " | ".join(summary_parts)
    return summary[:max_length] + "..." if len(summary) > max_length else summary


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    generation_start = time.time()
    
    try:
        # Input validation and processing
        if not request.model.strip():
            raise HTTPException(status_code=400, detail="Model must be specified")
        
        # Check for embedding model
        model_lower = request.model.lower()
        if 'embed' in model_lower and not any(word in model_lower for word in ['chat', 'instruct', 'code']):
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model}' is an embedding model. Please use a chat model."
            )
        
        user_id = request.user_id or get_or_create_user_id()
        
        # Session handling (optimized)
        if request.session_id and request.session_id in chat_sessions:
            session_id = request.session_id
            session = chat_sessions[session_id]
        else:
            session = create_new_session(request.model, user_id=user_id)
            session_id = session["session_id"]
            active_sessions[user_id] = session_id
        
        # Check limits
        if session["message_count"] >= settings.max_messages_per_session:
            raise HTTPException(
                status_code=429,
                detail=f"Maximum messages per session ({settings.max_messages_per_session}) exceeded."
            )
        
        # Add user message
        user_message_id = add_message_to_session(session_id, "user", request.message, request.model)
        update_session_title(session_id, request.message)
        
        # OPTIMIZED: Check cache first
        cache_key = get_cache_key(request.message, request.model, request.temperature)
        cached_response = get_cached_response(cache_key)
        
        if cached_response:
            # Use cached response
            assistant_message_id = add_message_to_session(session_id, "assistant", cached_response['content'], request.model)
            
            return ChatResponse(
                response=cached_response['content'],
                model=request.model,
                session_id=session_id,
                message_id=assistant_message_id,
                usage=cached_response.get('usage', 0),
                timestamp=datetime.now().isoformat(),
                cached=True
            )
        
        # ✅ FIXED: Build context with PROPER conversation history
        system_prompt = get_optimized_system_prompt(
            message=request.message, 
            model=request.model, 
            mode="chat"
        )
        messages = [{"role": "system", "content": system_prompt}]
        
        # ✅ CRITICAL FIX: Include COMPLETE conversation history properly
        # Get all messages from session (including the one we just added)
        all_session_messages = session["messages"]
        
        # Calculate how many messages we can fit within context limits
        total_context_length = len(system_prompt)
        context_limit = min(settings.max_context_length, 6000)  # Reasonable limit for gpt-oss-20b
        
        # ✅ NEW LOGIC: Include messages from newest to oldest, but maintain chronological order
        messages_to_include = []
        
        # Start from the most recent messages and work backwards
        for i in range(len(all_session_messages) - 1, -1, -1):
            msg = all_session_messages[i]
            msg_content = msg['content']
            msg_length = len(msg_content)
            
            # Check if we can fit this message
            if total_context_length + msg_length <= context_limit:
                messages_to_include.insert(0, msg)  # Insert at beginning to maintain chronological order
                total_context_length += msg_length
            else:
                break
            
            # Limit to last 20 messages maximum for performance
            if len(messages_to_include) >= 20:
                break
        
        # ✅ Add all included messages in chronological order
        for msg in messages_to_include:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        # Debug logging for gpt-oss-20b
        logger.info(f"Context for gpt-oss-20b: {len(messages)} messages, {total_context_length} chars")
        
        # OPTIMIZED: Ollama options for gpt-oss-20b
        options = {
            'temperature': request.temperature,
            'top_p': 0.9,
            'top_k': 40,
        }
        if request.max_tokens:
            options['num_predict'] = min(request.max_tokens, 2000)
        
        # Generate response with optimized retry logic
        max_retries = 2
        
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
                OLLAMA_ERRORS.inc()
                error_msg = str(e).lower()
                
                if attempt == max_retries - 1:
                    logger.error("Ollama request failed", model=request.model, error=str(e))
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to generate response: {str(e)[:100]}..."
                    )
                else:
                    # Quick retry without exponential backoff
                    await asyncio.sleep(0.5)
        
        # Process response
        if not response or 'message' not in response:
            raise HTTPException(status_code=500, detail="Invalid response from Ollama")
        
        assistant_content = response['message']['content']
        
        # Cache the response
        cache_data = {
            'content': assistant_content,
            'usage': response.get('eval_count', 0)
        }
        cache_response(cache_key, cache_data)
        
        # Add to session
        assistant_message_id = add_message_to_session(session_id, "assistant", assistant_content, request.model)
        
        generation_time = time.time() - generation_start
        RESPONSE_GENERATION_TIME.observe(generation_time)
        
        # ✅ Additional debug info for testing
        logger.info(f"Chat response generated in {generation_time:.2f}s for session {session_id}")
        
        return ChatResponse(
            response=assistant_content,
            model=request.model,
            session_id=session_id,
            message_id=assistant_message_id,
            usage=response.get('eval_count', 0),
            timestamp=datetime.now().isoformat(),
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)[:100]}...")

# OPTIMIZED: Session endpoints
@app.post("/api/sessions/new")
async def create_new_session_endpoint(request: NewSessionRequest):
    """Fast session creation"""
    try:
        session = create_new_session(request.model, request.title)
        return {"success": True, "session": session}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation error: {str(e)[:50]}...")

@app.get("/api/sessions")
async def get_sessions():
    """Optimized session retrieval"""
    try:
        # Quick list comprehension
        sessions = [
            SessionInfo(
                session_id=session_id,
                title=session["title"],
                model=session["model"],
                created_at=session["created_at"],
                updated_at=session["updated_at"],
                message_count=session["message_count"]
            ) for session_id, session in chat_sessions.items()
        ]
        
        # Sort by update time (faster than lambda)
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session retrieval error: {str(e)[:50]}...")

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Fast session deletion"""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Batch removal
        chat_sessions.pop(session_id, None)
        session_creation_times.pop(session_id, None)
        
        # Remove from active sessions
        for user_id, active_session_id in list(active_sessions.items()):
            if active_session_id == session_id:
                del active_sessions[user_id]
                break
        
        # Update metrics
        TOTAL_SESSIONS.set(len(chat_sessions))
        ACTIVE_SESSIONS.set(len(active_sessions))
        
        return {"success": True, "message": "Session deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion error: {str(e)[:50]}...")

@app.get("/api/models")
async def get_models():
    """Get only models that are actually available in Ollama"""
    try:
        # Check cache first
        cache_key = "available_models"
        cached_models = get_cached_response(cache_key)
        
        if cached_models:
            return cached_models
        
        # Get models from Ollama
        try:
            models_response = ollama.list()
        except Exception as e:
            logger.error("Failed to fetch models from Ollama", error=str(e))
            return {
                "models": [], 
                "error": "Ollama not available",
                "message": "Please ensure Ollama is running and accessible."
            }
        
        models = []
        
        # Extract models from response
        raw_models = getattr(models_response, 'models', None) or models_response.get('models', [])
        
        if not raw_models:
            return {
                "models": [], 
                "error": "No models available",
                "message": "No models found in Ollama. Please pull some models first."
            }
        
        # Filter out embedding models
        embedding_indicators = {'embed', 'embedding', 'nomic-embed', 'all-minilm', 'bge-', 'gte-', 'text-embedding'}
        
        for model in raw_models:
            # Extract model info
            model_name = getattr(model, 'model', None) or model.get('model', '')
            if not model_name:
                continue
            
            model_lower = model_name.lower()
            is_embedding = any(indicator in model_lower for indicator in embedding_indicators)
            is_chat_compatible = any(word in model_lower for word in ['chat', 'instruct', 'code'])
            
            # Only include non-embedding models OR models that explicitly support chat
            if not is_embedding or is_chat_compatible:
                models.append({
                    "name": model_name,
                    "size": getattr(model, 'size', None) or model.get('size', 0),
                    "size_gb": round((getattr(model, 'size', None) or model.get('size', 0)) / (1024**3), 2),
                    "type": "chat",
                    "warmed": model_name in model_warmup_cache,
                    "status": "available"
                })
        
        models.sort(key=lambda x: x['name'].lower())
        
        result = {"models": models, "total_models": len(models)}
        cache_response(cache_key, result)
        
        logger.info(f"Found {len(models)} available chat models")
        return result
        
    except Exception as e:
        logger.error("Models retrieval error", error=str(e))
        return {"models": [], "error": str(e)[:100]}

# OPTIMIZED: RAG endpoints with async processing
@app.post("/api/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Async document upload"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Async file processing
        file_info = await file_service.save_uploaded_file(file)
        
        # Process document asynchronously
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
            file_service.delete_file(file_info["file_id"], file_info["file_type"])
            raise HTTPException(status_code=400, detail=result["error"][:100])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)[:100]}...")

@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequest):
    query_start = time.time()
    
    try:
        # Check cache first
        cache_key = get_cache_key(request.query, request.model, request.temperature, "rag")
        cached_response = get_cached_response(cache_key)
        
        if cached_response:
            return RAGQueryResponse(
                response=cached_response['response'],
                query=request.query,
                model=request.model,
                context_used=cached_response['context_used'],
                timestamp=datetime.now().isoformat(),
                cached=True
            )
        
        # Get document context
        try:
            context_task = asyncio.wait_for(
                rag_service.get_document_context(request.query, max_chunks=request.max_chunks),
                timeout=5.0
            )
            context = await context_task
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Document search timeout")
        
        if not context:
            raise HTTPException(
                status_code=400, 
                detail="No documents available for RAG queries. Please upload documents first."
            )
        
        # ✅ ENHANCED: RAG with conversation awareness
        system_prompt = get_optimized_system_prompt(
            message=request.query,
            model=request.model,
            mode="rag",
            context=context
        )
        
        # ✅ NEW: Include recent conversation context in RAG mode
        # This helps maintain conversation flow even in RAG mode
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add a brief conversation context if available
        # (This is a simplified approach - you might want to get actual session context)
        messages.append({"role": "user", "content": request.query})
        
        # Fast Ollama generation with model-specific optimization
        context_limit = settings.get_context_limit_for_model(request.model)
        max_tokens = min(800, context_limit // 4)  # Adaptive response length
        
        options = {
            'temperature': request.temperature, 
            'num_predict': max_tokens,
            'top_p': 0.9,
            'top_k': 40
        }
        
        response = ollama.chat(model=request.model, messages=messages, options=options)
        
        # Get search results for context
        search_results = await rag_service.search_documents(request.query, top_k=request.max_chunks)
        
        # Cache the result
        cache_data = {
            'response': response['message']['content'],
            'context_used': search_results
        }
        cache_response(cache_key, cache_data)
        
        query_time = time.time() - query_start
        logger.info(f"Enhanced RAG query processed in {query_time:.2f}s for {request.model}")
        
        return RAGQueryResponse(
            response=response['message']['content'],
            query=request.query,
            model=request.model,
            context_used=search_results,
            timestamp=datetime.now().isoformat(),
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced RAG query failed: {str(e)[:100]}...")

@app.get("/api/rag/documents")
async def get_documents():
    """Fast document listing"""
    try:
        documents = rag_service.get_document_list()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)[:50]}...")

@app.delete("/api/rag/documents/{document_id}")
async def delete_document(document_id: str):
    """Async document deletion"""
    try:
        documents = rag_service.get_document_list()
        doc_info = next((doc for doc in documents if doc["id"] == document_id), None)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        success = rag_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")
        
        # Async file cleanup
        remaining_doc_ids = [doc["id"] for doc in documents if doc["id"] != document_id]
        file_service.cleanup_orphaned_files(remaining_doc_ids)
        
        return {"success": True, "message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)[:50]}...")

@app.get("/api/rag/stats")
async def get_rag_stats():
    """Cached RAG statistics"""
    try:
        cache_key = "rag_stats"
        cached_stats = get_cached_response(cache_key)
        
        if cached_stats:
            return cached_stats
        
        rag_stats = rag_service.get_document_stats()
        file_stats = file_service.get_storage_stats()
        
        result = {"rag_stats": rag_stats, "file_stats": file_stats}
        cache_response(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)[:50]}...")

@app.get("/api/rag/search")
async def search_documents(query: str, top_k: int = 5):
    """Fast document search"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Limit search parameters for speed
        top_k = min(top_k, 10)
        
        results = await rag_service.search_documents(query, top_k=top_k)
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)[:50]}...")

# OPTIMIZED: Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Fast metrics generation"""
    try:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Metrics generation failed")

# OPTIMIZED: Test endpoints
@app.get("/api/performance/cache-stats")
async def get_cache_stats():
    """Cache performance statistics"""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Debug endpoint not available")
    
    return {
        "cache_enabled": settings.enable_caching,
        "cached_responses": len(response_cache),
        "cache_size_mb": sum(len(str(data)) for data in response_cache.values()) / 1024 / 1024,
        "warmed_models": list(model_warmup_cache.keys()),
        "settings": {
            "cache_ttl": settings.cache_ttl,
            "max_context_length": settings.max_context_length,
            "chunk_batch_size": settings.chunk_batch_size
        }
    }

@app.get("/api/performance/benchmark")
async def benchmark_response():
    """Quick benchmark endpoint"""
    start = time.time()
    
    # Simulate some work
    await asyncio.sleep(0.001)
    
    end = time.time()
    
    return {
        "response_time_ms": round((end - start) * 1000, 2),
        "timestamp": datetime.now().isoformat(),
        "cache_enabled": settings.enable_caching,
        "environment": environment
    }

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

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Endpoint not found: {request.url.path}"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    
    config = {
        "app": "app:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": settings.debug,
        "log_level": "info" if not settings.debug else "debug",
        "access_log": False,  # Disable for performance
        "workers": 1 if settings.debug else 4,
    }
    
    logger.info("Starting Performance-Optimized Ollama Chat App", **config)
    uvicorn.run(**config)