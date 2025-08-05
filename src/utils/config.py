import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    
    # Neo4j
    neo4j_uri: str = Field(..., env="NEO4J_URI")
    neo4j_user: str = Field(..., env="NEO4J_USER")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    
    # AWS S3
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field("us-east-2", env="AWS_REGION")
    s3_bucket_name: str = Field(..., env="S3_BUCKET_NAME")
    
    # TTT Configuration
    ttt_cache_dir: str = Field("./data/adapters", env="TTT_CACHE_DIR")
    ttt_examples_dir: str = Field("./data/examples", env="TTT_EXAMPLES_DIR")
    ttt_learning_rate: float = Field(5e-5, env="TTT_LEARNING_RATE")
    ttt_num_epochs: int = Field(2, env="TTT_NUM_EPOCHS")
    ttt_lora_rank: int = Field(128, env="TTT_LORA_RANK")
    ttt_batch_size: int = Field(2, env="TTT_BATCH_SIZE")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    
    # Processing Configuration
    max_file_size_mb: int = Field(100, env="MAX_FILE_SIZE_MB")
    supported_extensions: list = Field(
        ['.pdf', '.txt', '.docx', '.doc', '.xlsx', '.xls', '.csv'],
        env="SUPPORTED_EXTENSIONS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

def get_data_dir(subdir: str = "") -> Path:
    """Get data directory path"""
    base_dir = Path("./data")
    if subdir:
        return base_dir / subdir
    return base_dir

def ensure_data_dirs():
    """Ensure all required data directories exist"""
    dirs = [
        get_data_dir("adapters"),
        get_data_dir("examples"),
        get_data_dir("downloads"),
        get_data_dir("processed"),
        get_data_dir("logs")
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs