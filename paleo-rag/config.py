"""
Configuration module for Paleo RAG system.

Loads environment variables from .env file and provides typed access
to all configuration parameters using pydantic-settings.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Configuration settings for the Paleo RAG system.
    
    All settings can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths
    DATA_DIR: Path = Path("./data")
    INDEX_DIR: Path = Path("./index")
    RAW_DATA_FILE: Path = Path("./data/raw_synthetic.jsonl")
    FAISS_INDEX_PATH: Path = Path("./index/faiss_index")

    # Models - using locally cached model name format
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L6-v2"
    LLM_MODEL: str = "qwen2.5:7b"
    LLM_BASE_URL: str = "http://localhost:11434"

    # Retrieval parameters
    TOP_K: int = 5
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50

    # Generation parameters
    NUM_DOCUMENTS: int = 200
    RANDOM_SEED: int = 42

    # LLM parameters
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 512

    # Evaluation
    EVAL_NUM_QUESTIONS: int = 10

    def __post_init__(self) -> None:
        """Ensure directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
