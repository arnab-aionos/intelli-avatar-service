"""
Configuration management for IntelliAvatar Service.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    # Standard OpenAI (optional - if using Azure OpenAI)
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Azure OpenAI (using Azure Foundry)
    azure_openai_key: str = Field(..., env="AZURE_OPENAI_KEY")
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field(default="gpt-4o", env="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-02-01", env="AZURE_OPENAI_API_VERSION")
    
    # Other API keys
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    daily_api_key: str = Field(default="", env="DAILY_API_KEY")  # Optional now
    
    # LiveKit Configuration
    livekit_url: str = Field(..., env="LIVEKIT_URL")
    livekit_api_key: str = Field(..., env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(..., env="LIVEKIT_API_SECRET")
    
    # Server Configuration
    server_host: str = Field(default="0.0.0.0", env="SERVER_HOST")
    server_port: int = Field(default=8000, env="SERVER_PORT")
    api_port: int = Field(default=8001, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # WebRTC Configuration
    webrtc_stun_server: str = Field(
        default="stun:stun.l.google.com:19302",
        env="WEBRTC_STUN_SERVER"
    )
    webrtc_turn_server: str = Field(default="", env="WEBRTC_TURN_SERVER")
    webrtc_turn_username: str = Field(default="", env="WEBRTC_TURN_USERNAME")
    webrtc_turn_password: str = Field(default="", env="WEBRTC_TURN_PASSWORD")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/intelliavatar",
        env="DATABASE_URL"
    )
    
    # MuseTalk Configuration
    musetalk_models_path: str = Field(
        default="../talkingavatar/Musetalk/models",
        env="MUSETALK_MODELS_PATH"
    )
    musetalk_device: str = Field(default="cuda", env="MUSETALK_DEVICE")
    musetalk_fps: int = Field(default=25, env="MUSETALK_FPS")
    musetalk_batch_size: int = Field(default=4, env="MUSETALK_BATCH_SIZE")
    
    # Authentication
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/server.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def musetalk_models_full_path(self) -> Path:
        """Get absolute path to MuseTalk models."""
        path = Path(self.musetalk_models_path)
        if not path.is_absolute():
            # Make relative to project root
            project_root = Path(__file__).parent
            path = (project_root / path).resolve()
        return path


# Global settings instance
settings = Settings()


# Ensure log directory exists
log_dir = Path(settings.log_file).parent
log_dir.mkdir(exist_ok=True, parents=True)
