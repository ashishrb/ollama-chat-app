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
    max_messages_per_session: int = 1000
    session_cleanup_interval: int = 3600  # 1 hour
    session_ttl: int = 86400 * 7  # 7 days
    
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
    
    # Performance & Caching Configuration
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_context_length: int = 4000  # Maximum context length for prompts
    
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
        
        print(f"🛠️ Development config applied - CORS: {self.cors_origins}")

# Create the global settings instance
settings = Settings()

# Print configuration summary
print(f"""
=== OLLAMA CHAT APP CONFIGURATION ===
Environment: {os.getenv('ENVIRONMENT', 'development')}
Debug Mode: {settings.debug}
CORS Origins: {settings.cors_origins}
Trusted Hosts: {settings.trusted_hosts}
Rate Limit: {settings.rate_limit_requests} req/min
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
        print(f"✅ Ollama connected - {len(models.get('models', []))} models available")
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

# Run validations if this file is executed directly
if __name__ == "__main__":
    print("Running configuration validation...")
    validate_ollama_connection()
    validate_upload_directory()
    print("Configuration validation complete.")