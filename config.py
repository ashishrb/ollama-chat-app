import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Configuration
    app_name: str = "Ollama Chat App"
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
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 1 minute
    
    # Security
    cors_origins: list = ["*"]
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
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    settings.debug = False
    settings.cors_origins = os.getenv("CORS_ORIGINS", "https://yourdomain.com").split(",")
    settings.enable_api_key_auth = True
elif os.getenv("ENVIRONMENT") == "staging":
    settings.debug = False
    settings.cors_origins = ["https://staging.yourdomain.com"]
elif os.getenv("ENVIRONMENT") == "development":
    settings.debug = True
    settings.cors_origins = ["*"]
