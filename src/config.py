"""
Configuration management for pharmaceutical research assistant.
Uses environment variables with sensible defaults for development.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="sqlite:///./pharmaceutical_research.db",
        description="Database connection URL"
    )
    echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging"
    )
    pool_size: int = Field(
        default=5,
        description="Database connection pool size"
    )


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    expire_time: int = Field(default=3600, description="Default cache expiration in seconds")


class EmbeddingSettings(BaseSettings):
    """Text embedding model configuration."""
    
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model name"
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum token length for embeddings"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )


class APISettings(BaseSettings):
    """External API configuration."""
    
    pubmed_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        description="PubMed E-utilities base URL"
    )
    pubmed_email: str = Field(
        default="research@pharmaceutical-ai.demo",
        description="Email for PubMed API requests"
    )
    pubmed_api_key: Optional[str] = Field(
        default=None,
        description="PubMed API key for increased rate limits"
    )
    request_timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds"
    )


class AppSettings(BaseSettings):
    """Main application settings."""
    
    # Application
    app_name: str = Field(
        default="Pharmaceutical Research Assistant",
        description="Application name"
    )
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # API
    api_prefix: str = Field(default="/api/v1", description="API path prefix")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # Security
    secret_key: str = Field(
        default="demo-secret-key-change-in-production",
        description="Secret key for session management"
    )
    
    # Performance
    max_concurrent_requests: int = Field(
        default=100,
        description="Maximum concurrent API requests"
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Rate limit per minute per IP"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
    default="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
    description="Log message format"
)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class Settings(BaseSettings):
    """Combined application settings."""
    
    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    api: APISettings = APISettings()

    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL."""
        auth = f":{self.redis.password}@" if self.redis.password else ""
        return f"redis://{auth}{self.redis.host}:{self.redis.port}/{self.redis.db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings