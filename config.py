import os
from typing import Optional, List, Union
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Environment Configuration
    environment: str = "development"
    
    # App Configuration
    app_name: str = "Ollama Chat App with RAG"
    app_version: str = "2.0.0"
    debug: bool = True  # Default to development mode
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 30
    ollama_max_retries: int = 3
    
    # Session Management
    max_sessions_per_user: int = 100
    max_messages_per_session: int = 2000  # ✅ Enhanced: Increased from 1000
    session_cleanup_interval: int = 3600  # 1 hour
    session_ttl: int = 86400 * 14  # ✅ Enhanced: 14 days (from 7 days)
    
    # Enhanced Rate Limiting (more permissive for development)
    rate_limit_requests: int = 300  # Increased from 100
    rate_limit_window: int = 60  # 1 minute
    
    # Security & CORS - More flexible configuration
    cors_origins: Union[List[str], str] = ["*"]
    trusted_hosts: List[str] = ["*"]  # Allow all hosts in development
    api_key_header: str = "X-API-Key"
    enable_api_key_auth: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Database (for production)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8001
    
    # File Upload Configuration
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    upload_dir: str = "uploads"
    
    # RAG Configuration
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_embedding_model: str = "nomic-embed-text:v1.5"
    
    # ✅ ENHANCED: Performance & Caching Configuration for gpt-oss-20b
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    
    # ✅ ENHANCED: Better context management for conversation memory
    max_context_length: int = 12000  # ✅ Increased from 4000 for better memory
    max_conversation_messages: int = 30  # ✅ Up from 20 for longer conversations
    context_overlap_buffer: int = 1000  # ✅ Buffer for context management
    chunk_batch_size: int = 5  # Parallel processing
    
    # ✅ NEW: Smart context truncation settings
    enable_smart_truncation: bool = True
    preserve_system_prompt: bool = True
    preserve_recent_messages: int = 6  # Always keep last 6 messages
    
    # ✅ NEW: Model-specific optimizations for different model sizes
    gpt_oss_20b_context_limit: int = 16000  # Specific limit for gpt-oss-20b
    small_model_context_limit: int = 8000   # For smaller models (3B-7B)
    medium_model_context_limit: int = 12000 # For medium models (13B-15B)
    large_model_context_limit: int = 20000  # For larger models (70B+)
    
    # ✅ NEW: Advanced conversation management
    enable_conversation_summary: bool = True
    summary_trigger_messages: int = 25  # Start summarizing after 25 messages
    max_summary_length: int = 300  # Maximum summary length
    
    # ✅ NEW: Performance tuning for different scenarios
    fast_response_mode: bool = False  # Quick responses vs quality
    adaptive_context_sizing: bool = True  # Adjust context based on model
    context_compression_ratio: float = 0.8  # How much to compress long contexts
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra environment variables
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._apply_environment_overrides()
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        environment = self.environment.lower()
        
        if environment == "production":
            self._apply_production_config()
        elif environment == "staging":
            self._apply_staging_config()
        elif environment == "development":
            self._apply_development_config()
        else:
            # Default to development
            self._apply_development_config()
    
    def _apply_production_config(self):
        """Production environment configuration"""
        self.debug = False
        
        # Strict CORS in production
        cors_env = os.getenv("CORS_ORIGINS")
        if cors_env:
            self.cors_origins = [origin.strip() for origin in cors_env.split(",")]
        else:
            self.cors_origins = ["https://yourdomain.com"]
        
        # Strict trusted hosts in production
        hosts_env = os.getenv("ALLOWED_HOSTS")
        if hosts_env:
            self.trusted_hosts = [host.strip() for host in hosts_env.split(",")]
        else:
            self.trusted_hosts = ["yourdomain.com", "www.yourdomain.com", "localhost", "127.0.0.1"]
        
        # Enable authentication in production
        self.enable_api_key_auth = True
        
        # More conservative rate limiting in production
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        
        # ✅ Production-optimized context settings
        self.fast_response_mode = True  # Prioritize speed in production
        self.cache_ttl = 600  # Longer cache in production (10 minutes)
        
        print(f"🚀 Production config applied - CORS: {self.cors_origins}")
    
    def _apply_staging_config(self):
        """Staging environment configuration"""
        self.debug = False
        self.cors_origins = ["https://staging.yourdomain.com", "http://localhost:3000"]
        self.trusted_hosts = ["staging.yourdomain.com", "localhost", "127.0.0.1"]
        self.enable_api_key_auth = False  # Optional in staging
        
        print(f"🧪 Staging config applied - CORS: {self.cors_origins}")
    
    def _apply_development_config(self):
        """Development environment configuration"""
        self.debug = True
        
        # Very permissive CORS for development
        cors_env = os.getenv("CORS_ORIGINS")
        if cors_env and cors_env != "*":
            self.cors_origins = [origin.strip() for origin in cors_env.split(",")]
        else:
            self.cors_origins = ["*"]
        
        # Allow all hosts in development
        self.trusted_hosts = ["*"]
        
        # No authentication in development
        self.enable_api_key_auth = False
        
        # More lenient rate limiting for development
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "300"))
        
        # ✅ Development-optimized settings
        self.fast_response_mode = False  # Quality over speed in development
        self.enable_smart_truncation = True
        self.adaptive_context_sizing = True
        
        print(f"🛠️ Development config applied - CORS: {self.cors_origins}")

    # ✅ NEW: Model-specific context management methods
    def get_context_limit_for_model(self, model_name: str) -> int:
        """
        Get appropriate context limit based on model size and type
        Optimized for gpt-oss-20b and other models
        """
        model_lower = model_name.lower()
        
        # ✅ gpt-oss-20b specific optimization (your primary model)
        if 'gpt-oss' in model_lower and '20b' in model_lower:
            return self.gpt_oss_20b_context_limit
        
        # Other GPT-OSS variants
        elif 'gpt-oss' in model_lower:
            return self.medium_model_context_limit
            
        # Large models (70B+, 72B, etc.)
        elif any(size in model_lower for size in ['70b', '72b', '65b', '80b']):
            return self.large_model_context_limit
            
        # Medium models (13B-20B range)
        elif any(size in model_lower for size in ['13b', '14b', '15b', '16b', '18b', '20b']):
            return self.medium_model_context_limit
            
        # Small models (7B and below)
        elif any(size in model_lower for size in ['3b', '7b', '8b', '1b', '2b']):
            return self.small_model_context_limit
            
        # Code-specific models (usually need more context)
        elif any(word in model_lower for word in ['code', 'codellama', 'coder']):
            return self.medium_model_context_limit
            
        # Embedding models (minimal context needed)
        elif any(word in model_lower for word in ['embed', 'embedding', 'nomic-embed']):
            return 2000  # Minimal context for embedding models
            
        # Default to medium for unknown models
        else:
            return self.medium_model_context_limit
    
    def get_optimal_message_count(self, model_name: str) -> int:
        """Get optimal number of messages to include based on model"""
        context_limit = self.get_context_limit_for_model(model_name)
        
        # Estimate messages based on context limit
        # Average message length ~200 chars
        avg_message_length = 200
        estimated_messages = context_limit // avg_message_length
        
        # Cap at reasonable limits
        return min(estimated_messages, self.max_conversation_messages)
    
    def should_use_smart_truncation(self, model_name: str, message_count: int) -> bool:
        """Determine if smart truncation should be used"""
        if not self.enable_smart_truncation:
            return False
        
        # Always use smart truncation for gpt-oss-20b
        if 'gpt-oss' in model_name.lower() and '20b' in model_name.lower():
            return True
            
        # Use for longer conversations
        return message_count > 10
    
    def get_cache_ttl_for_model(self, model_name: str) -> int:
        """Get appropriate cache TTL based on model type"""
        model_lower = model_name.lower()
        
        # Longer cache for slower/larger models
        if any(size in model_lower for size in ['70b', '72b', '65b']):
            return self.cache_ttl * 2  # Double cache time for large models
        elif 'gpt-oss' in model_lower and '20b' in model_lower:
            return int(self.cache_ttl * 1.5)  # 1.5x cache for gpt-oss-20b
        else:
            return self.cache_ttl
    
    def get_performance_mode(self) -> str:
        """Get current performance mode"""
        if self.fast_response_mode:
            return "fast"
        elif self.environment == "production":
            return "balanced"
        else:
            return "quality"
    
    def get_context_summary(self) -> dict:
        """Get a summary of current context configuration"""
        return {
            "max_context_length": self.max_context_length,
            "max_conversation_messages": self.max_conversation_messages,
            "preserve_recent_messages": self.preserve_recent_messages,
            "smart_truncation_enabled": self.enable_smart_truncation,
            "conversation_summary_enabled": self.enable_conversation_summary,
            "adaptive_context_sizing": self.adaptive_context_sizing,
            "performance_mode": self.get_performance_mode(),
            "model_specific_limits": {
                "gpt_oss_20b": self.gpt_oss_20b_context_limit,
                "small_models": self.small_model_context_limit,
                "medium_models": self.medium_model_context_limit,
                "large_models": self.large_model_context_limit
            }
        }

# Create the global settings instance
settings = Settings()

# Print enhanced configuration summary
print(f"""
=== ENHANCED OLLAMA CHAT APP CONFIGURATION ===
Environment: {os.getenv('ENVIRONMENT', 'development')}
Debug Mode: {settings.debug}
CORS Origins: {settings.cors_origins}
Trusted Hosts: {settings.trusted_hosts}
Rate Limit: {settings.rate_limit_requests} req/min

=== CONTEXT MANAGEMENT SETTINGS ===
Max Context Length: {settings.max_context_length} chars
Max Conversation Messages: {settings.max_conversation_messages}
Smart Truncation: {settings.enable_smart_truncation}
Preserve Recent Messages: {settings.preserve_recent_messages}

=== MODEL-SPECIFIC LIMITS ===
GPT-OSS-20B: {settings.gpt_oss_20b_context_limit} chars
Small Models: {settings.small_model_context_limit} chars
Medium Models: {settings.medium_model_context_limit} chars
Large Models: {settings.large_model_context_limit} chars

=== PERFORMANCE SETTINGS ===
Caching Enabled: {settings.enable_caching}
Cache TTL: {settings.cache_ttl}s
Performance Mode: {settings.get_performance_mode()}
Adaptive Context: {settings.adaptive_context_sizing}

Ollama Host: {settings.ollama_host}
Upload Dir: {settings.upload_dir}
Max Upload Size: {settings.max_upload_size / (1024*1024):.0f}MB
=====================================
""")

# Validation functions
def validate_ollama_connection():
    """Validate Ollama connection"""
    try:
        import ollama
        models = ollama.list()
        model_count = len(models.get('models', []))
        print(f"✅ Ollama connected - {model_count} models available")
        
        # Check for gpt-oss-20b specifically
        model_names = [model.get('model', '') for model in models.get('models', [])]
        gpt_oss_models = [name for name in model_names if 'gpt-oss' in name.lower()]
        if gpt_oss_models:
            print(f"✅ Found GPT-OSS models: {', '.join(gpt_oss_models)}")
        else:
            print("⚠️ No GPT-OSS models found. You may need to pull gpt-oss:20B")
            
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def validate_upload_directory():
    """Ensure upload directory exists"""
    try:
        os.makedirs(settings.upload_dir, exist_ok=True)
        print(f"✅ Upload directory ready: {settings.upload_dir}")
        return True
    except Exception as e:
        print(f"❌ Upload directory creation failed: {e}")
        return False

def validate_context_settings():
    """Validate context management settings"""
    try:
        # Test model-specific context limits
        test_models = ['gpt-oss:20B', 'llama3.2:3b', 'llama3.2:70b', 'codellama:7b']
        
        print("✅ Context limits validation:")
        for model in test_models:
            limit = settings.get_context_limit_for_model(model)
            msg_count = settings.get_optimal_message_count(model)
            print(f"  {model}: {limit} chars, ~{msg_count} messages")
        
        return True
    except Exception as e:
        print(f"❌ Context settings validation failed: {e}")
        return False

# Run validations if this file is executed directly
if __name__ == "__main__":
    print("Running enhanced configuration validation...")
    validate_ollama_connection()
    validate_upload_directory()
    validate_context_settings()
    print("Enhanced configuration validation complete.")
    
    # Print context summary
    print("\n=== CONTEXT CONFIGURATION SUMMARY ===")
    import json
    print(json.dumps(settings.get_context_summary(), indent=2))