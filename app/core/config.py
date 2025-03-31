from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG System"
    
    # Model Settings
    MODEL_TYPE: str = "self_hosted"  # or "openai"
    MODEL_URL: str  # URL for remote LLM service
    MODEL_NAME: str  # Name of the model to use
    MODEL_ENDPOINT: str = "/generate"  # Endpoint for LLM generation
    EMBEDDING_MODEL_URL: str  # URL for remote embedding service
    EMBEDDING_MODEL_NAME: str  # Name of the embedding model to use
    EMBEDDING_ENDPOINT: str = "/embed"  # Endpoint for single embedding
    EMBEDDING_BATCH_ENDPOINT: str = "/embed_batch"  # Endpoint for batch embeddings
    
    # Optional API keys for remote services if needed
    REMOTE_LLM_API_KEY: Optional[str] = None
    REMOTE_EMBEDDING_API_KEY: Optional[str] = None
    
    # Vector DB Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "documents"
    
    # Redis Settings (for caching)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Search Settings
    TOP_K_RESULTS: int = 5
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings() 