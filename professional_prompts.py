# professional_prompts.py - Optimized System Prompts for Speed and Quality

import os


class SystemPrompts:
    """Collection of optimized system prompts for different use cases"""
    
    # ULTRA-FAST: Minimal prompt for maximum speed
    MINIMAL = "You are a helpful AI assistant. Be direct and concise."
    
    # BALANCED: Good balance of quality and speed
    BALANCED = ("You are a professional AI assistant. Provide clear, direct answers. "
               "Use proper markdown for code. Be concise but helpful.")
    
    # CODING: Optimized for programming tasks
    CODING = ("You are a coding assistant. Provide complete, working code with brief explanations. "
             "Use ```language``` blocks for code. No verbose explanations.")
    
    # RAG: Optimized for document-based queries
    RAG = ("Answer questions using only the provided context. "
          "If information isn't in the context, say 'Not found in provided documents.'")
    
    # PROFESSIONAL: For business/professional contexts
    PROFESSIONAL = ("You are a professional business assistant. Provide accurate, "
                   "well-structured responses. Be formal but approachable.")
    
    # CREATIVE: For creative tasks
    CREATIVE = ("You are a creative assistant. Be imaginative and helpful. "
               "Provide original, engaging content.")
    
    @classmethod
    def get_prompt(cls, mode: str = "balanced") -> str:
        """Get system prompt based on mode"""
        prompts = {
            "minimal": cls.MINIMAL,
            "balanced": cls.BALANCED,
            "coding": cls.CODING,
            "rag": cls.RAG,
            "professional": cls.PROFESSIONAL,
            "creative": cls.CREATIVE
        }
        return prompts.get(mode.lower(), cls.BALANCED)
    
    @classmethod
    def get_rag_prompt(cls, context: str, max_context_length: int = 2000) -> str:
        """Generate RAG-specific prompt with context"""
        # Truncate context if too long for speed
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        return f"{cls.RAG}\n\nContext:\n{context}\n\nQuestion:"

# Performance-optimized prompt templates
class FastPrompts:
    """Ultra-fast prompts for high-throughput scenarios"""
    
    # Extremely minimal prompts
    CHAT = "Be helpful and direct."
    CODE = "Provide working code."
    SEARCH = "Answer from context only."
    SUMMARY = "Summarize briefly."
    
    # Template functions for dynamic prompts
    @staticmethod
    def chat_prompt(user_type: str = "user") -> str:
        """Ultra-minimal chat prompt"""
        return f"Help the {user_type} directly and concisely."
    
    @staticmethod
    def code_prompt(language: str = "python") -> str:
        """Language-specific code prompt"""
        return f"Provide working {language} code with minimal explanation."
    
    @staticmethod
    def rag_prompt(context: str) -> str:
        """Minimal RAG prompt"""
        # Super short context for speed
        short_context = context[:500] + "..." if len(context) > 500 else context
        return f"Context: {short_context}\nAnswer from context:"

# Quality-focused prompts (slower but better results)
class QualityPrompts:
    """High-quality prompts for when response quality is more important than speed"""
    
    EXPERT_CHAT = """You are an expert AI assistant with deep knowledge across multiple domains. 
    Provide thoughtful, accurate, and well-structured responses. When discussing technical topics, 
    ensure precision. When coding, provide complete, production-ready solutions with proper error 
    handling and best practices. Always consider the context and provide the most helpful response possible."""
    
    EXPERT_CODE = """You are a senior software engineer and architect. Provide production-quality code that:
    1. Follows best practices and design patterns
    2. Includes proper error handling and edge case management
    3. Is well-documented with clear comments
    4. Uses appropriate data structures and algorithms
    5. Considers security, performance, and maintainability
    
    Format code with proper markdown and explain key decisions."""
    
    EXPERT_RAG = """You are a research assistant with access to specific documents. Your task is to:
    1. Carefully analyze the provided context
    2. Extract relevant information that directly answers the question
    3. Synthesize information from multiple sources if needed
    4. Clearly distinguish between what's in the documents vs. general knowledge
    5. Provide accurate, well-sourced answers
    
    If the question cannot be answered from the provided context, clearly state this."""
    
    @staticmethod
    def expert_rag_prompt(context: str) -> str:
        """High-quality RAG prompt with full context"""
        return f"{QualityPrompts.EXPERT_RAG}\n\nDocument Context:\n{context}\n\nQuestion:"

# Dynamic prompt selector based on performance requirements
class SmartPrompts:
    """Intelligently select prompts based on performance requirements"""
    
    def __init__(self, prefer_speed: bool = True):
        self.prefer_speed = prefer_speed
        self.fast_prompts = FastPrompts()
        self.quality_prompts = QualityPrompts()
        self.system_prompts = SystemPrompts()
    
    def get_chat_prompt(self, context_length: int = 0, model_size: str = "medium") -> str:
        """Select chat prompt based on context and model"""
        if self.prefer_speed or context_length > 2000 or model_size == "small":
            return self.fast_prompts.CHAT
        elif model_size == "large":
            return self.quality_prompts.EXPERT_CHAT
        else:
            return self.system_prompts.BALANCED
    
    def get_code_prompt(self, language: str = "python", complexity: str = "simple") -> str:
        """Select code prompt based on requirements"""
        if self.prefer_speed or complexity == "simple":
            return self.fast_prompts.code_prompt(language)
        else:
            return self.quality_prompts.EXPERT_CODE
    
    def get_rag_prompt(self, context: str, quality_mode: bool = False) -> str:
        """Select RAG prompt based on quality requirements"""
        if self.prefer_speed and not quality_mode:
            return self.fast_prompts.rag_prompt(context)
        elif quality_mode:
            return self.quality_prompts.expert_rag_prompt(context)
        else:
            return self.system_prompts.get_rag_prompt(context)

# Usage in app.py - replace the get_optimized_system_prompt function:

def get_optimized_system_prompt(mode: str = "balanced", model_size: str = "medium", prefer_speed: bool = True) -> str:
    """Get optimized system prompt based on requirements"""
    
    # Initialize smart prompt selector
    smart_prompts = SmartPrompts(prefer_speed=prefer_speed)
    
    # Select prompt based on mode
    if mode == "chat":
        return smart_prompts.get_chat_prompt(model_size=model_size)
    elif mode == "coding":
        return smart_prompts.get_code_prompt(complexity="simple" if prefer_speed else "complex")
    elif mode == "minimal":
        return FastPrompts.CHAT
    elif mode == "quality":
        return QualityPrompts.EXPERT_CHAT
    else:
        return SystemPrompts.get_prompt(mode)

# Performance testing prompts
class BenchmarkPrompts:
    """Prompts for testing and benchmarking performance"""
    
    # Ultra-minimal for speed testing
    SPEED_TEST = "Hi"
    
    # Standard benchmark
    STANDARD = "You are a helpful assistant."
    
    # Heavy prompt for stress testing
    STRESS_TEST = """You are an advanced AI assistant with comprehensive knowledge across multiple domains including but not limited to: computer science, mathematics, physics, chemistry, biology, history, literature, philosophy, psychology, economics, business, arts, and technology. Your primary objectives are to provide accurate, helpful, and well-structured responses while maintaining professional standards. When addressing technical questions, ensure precision and clarity. For coding problems, provide complete, production-ready solutions with proper error handling, documentation, and best practices. Always consider the context of the question and provide the most relevant and useful information possible."""

# Example usage functions for integration
def get_prompt_for_request(message: str, model: str, mode: str = "auto") -> str:
    """Automatically select the best prompt based on request characteristics"""
    
    message_length = len(message)
    is_code_request = any(keyword in message.lower() for keyword in ['code', 'function', 'class', 'algorithm', 'program'])
    is_short_query = message_length < 50
    
    # Determine model size from name
    model_lower = model.lower()
    if any(size in model_lower for size in ['3b', '7b', 'small']):
        model_size = "small"
    elif any(size in model_lower for size in ['13b', '20b', 'medium']):
        model_size = "medium"
    else:
        model_size = "large"
    
    # Auto-select mode if not specified
    if mode == "auto":
        if is_code_request:
            mode = "coding"
        elif is_short_query:
            mode = "minimal"
        else:
            mode = "balanced"
    
    # Prefer speed for smaller models and shorter queries
    prefer_speed = model_size == "small" or is_short_query
    
    return get_optimized_system_prompt(mode, model_size, prefer_speed)

# Configuration for different deployment scenarios
class PromptConfig:
    """Configuration for different deployment scenarios"""
    
    # Development: Fast iteration
    DEVELOPMENT = {
        "default_mode": "minimal",
        "prefer_speed": True,
        "cache_prompts": True,
        "max_context_length": 1000
    }
    
    # Production: Balance of speed and quality
    PRODUCTION = {
        "default_mode": "balanced",
        "prefer_speed": True,
        "cache_prompts": True,
        "max_context_length": 2000
    }
    
    # Quality: Best responses regardless of speed
    QUALITY = {
        "default_mode": "quality",
        "prefer_speed": False,
        "cache_prompts": False,
        "max_context_length": 4000
    }
    
    @classmethod
    def get_config(cls, environment: str = "production"):
        """Get configuration for environment"""
        configs = {
            "development": cls.DEVELOPMENT,
            "production": cls.PRODUCTION,
            "quality": cls.QUALITY
        }
        return configs.get(environment, cls.PRODUCTION)

# Integration example for the FastAPI app
def integrate_with_fastapi_app():
    """Example of how to integrate with the FastAPI app"""
    
    # In app.py, replace get_optimized_system_prompt with:
    
    def get_system_prompt_for_request(request, model: str) -> str:
        """Get system prompt optimized for the specific request"""
        
        # Get environment configuration
        environment = os.getenv("ENVIRONMENT", "production")
        config = PromptConfig.get_config(environment)
        
        # Determine prompt based on request
        if hasattr(request, 'message'):
            return get_prompt_for_request(
                request.message, 
                model, 
                config.get("default_mode", "balanced")
            )
        else:
            return SystemPrompts.get_prompt(config.get("default_mode", "balanced"))
    
    return get_system_prompt_for_request