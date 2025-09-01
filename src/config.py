#!/usr/bin/env python3
"""
Configuration Module

Uses pydantic-settings for robust configuration management from environment variables.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fix tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Streamlit import for utilities (conditional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


class ModelConfig(BaseModel):
    """Model configuration settings"""
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    claude_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 2000


class ProcessingConfig(BaseModel):
    """Processing configuration settings"""
    batch_size: int = 20
    description_batch_size: int = 25
    max_workers: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.35
    relevancy_threshold: float = 0.5
    primary_threshold: float = 0.6
    min_display_threshold: float = 0.15
    supported_file_extensions: List[str] = ['.pdf', '.docx', '.doc', '.txt', '.md']
    faiss_store_name: str = "default"
    skip_processed_files: bool = True


class UIConfig(BaseModel):
    """UI configuration settings"""
    page_title: str = "AI Due Diligence"
    page_icon: str = "🤖"
    layout: str = "wide"
    top_k_search_results: int = 5


class PathsConfig(BaseModel):
    """Paths configuration with computed properties"""
    data_dir: str = "data"
    checklist_dir: str = "data/checklist"
    questions_dir: str = "data/questions"
    strategy_dir: str = "data/strategy"
    vdrs_dir: str = "data/vdrs"
    faiss_dir: str = "data/enhanced_faiss"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def checklist_path(self) -> Path:
        return Path(self.checklist_dir)

    @property
    def questions_path(self) -> Path:
        return Path(self.questions_dir)

    @property
    def strategy_path(self) -> Path:
        return Path(self.strategy_dir)
    
    @property
    def vdrs_path(self) -> Path:
        return Path(self.vdrs_dir)
    
    @property
    def faiss_path(self) -> Path:
        return Path(self.faiss_dir)


class APIConfig(BaseModel):
    """API configuration settings"""
    anthropic_api_key: Optional[str] = None
    max_concurrent_requests: int = 10


class Config(BaseSettings):
    """Main application configuration using pydantic-settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"  # Allow extra environment variables to be ignored
    )
    
    # Model settings
    sentence_transformer_model: str = Field(default="all-MiniLM-L6-v2", env="SENTENCE_TRANSFORMER_MODEL")
    claude_model: str = Field(default="claude-sonnet-4-20250514", env="CLAUDE_MODEL")
    temperature: float = Field(default=0.3, env="CLAUDE_TEMPERATURE")
    max_tokens: int = Field(default=2000, env="CLAUDE_MAX_TOKENS")
    
    # Processing settings (optimized for large datasets)
    batch_size: int = Field(default=20, env="BATCH_SIZE")
    description_batch_size: int = Field(default=25, env="DESCRIPTION_BATCH_SIZE") 
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    similarity_threshold: float = Field(default=0.35, env="SIMILARITY_THRESHOLD")
    relevancy_threshold: float = Field(default=0.5, env="RELEVANCY_THRESHOLD")
    primary_threshold: float = Field(default=0.6, env="PRIMARY_THRESHOLD")
    min_display_threshold: float = Field(default=0.15, env="MIN_DISPLAY_THRESHOLD")
    supported_file_extensions: List[str] = Field(
        default=['.pdf', '.docx', '.doc', '.txt', '.md'], 
        env="SUPPORTED_FILE_EXTENSIONS"
    )
    faiss_store_name: str = Field(default="default", env="FAISS_STORE_NAME")
    skip_processed_files: bool = Field(default=True, env="SKIP_PROCESSED_FILES")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    suppress_langchain_warnings: bool = Field(default=True, env="SUPPRESS_LANGCHAIN_WARNINGS")
    
    # UI settings
    page_title: str = Field(default="AI Due Diligence", env="PAGE_TITLE")
    page_icon: str = Field(default="🤖", env="PAGE_ICON")
    layout: str = Field(default="wide", env="LAYOUT")
    top_k_search_results: int = Field(default=5, env="TOP_K_SEARCH_RESULTS")
    
    # Path settings
    data_dir: str = Field(default="data", env="DATA_DIR")
    checklist_dir: str = Field(default="data/checklist", env="CHECKLIST_DIR")
    questions_dir: str = Field(default="data/questions", env="QUESTIONS_DIR")
    strategy_dir: str = Field(default="data/strategy", env="STRATEGY_DIR")
    vdrs_dir: str = Field(default="data/vdrs", env="VDRS_DIR")
    faiss_dir: str = Field(default="data/enhanced_faiss", env="FAISS_DIR")
    
    # API settings
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    @property
    def model(self) -> ModelConfig:
        """Get model configuration"""
        return ModelConfig(
            sentence_transformer_model=self.sentence_transformer_model,
            claude_model=self.claude_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    @property
    def processing(self) -> ProcessingConfig:
        """Get processing configuration"""
        return ProcessingConfig(
            batch_size=self.batch_size,
            description_batch_size=self.description_batch_size,
            max_workers=self.max_workers,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            similarity_threshold=self.similarity_threshold,
            relevancy_threshold=self.relevancy_threshold,
            primary_threshold=self.primary_threshold,
            min_display_threshold=self.min_display_threshold,
            supported_file_extensions=self.supported_file_extensions,
            faiss_store_name=self.faiss_store_name,
            skip_processed_files=self.skip_processed_files
        )
    
    @property
    def ui(self) -> UIConfig:
        """Get UI configuration"""
        return UIConfig(
            page_title=self.page_title,
            page_icon=self.page_icon,
            layout=self.layout,
            top_k_search_results=self.top_k_search_results
        )
    
    @property
    def paths(self) -> PathsConfig:
        """Get paths configuration"""
        return PathsConfig(
            data_dir=self.data_dir,
            checklist_dir=self.checklist_dir,
            questions_dir=self.questions_dir,
            strategy_dir=self.strategy_dir,
            vdrs_dir=self.vdrs_dir,
            faiss_dir=self.faiss_dir
        )
    
    @property
    def api(self) -> APIConfig:
        """Get API configuration"""
        return APIConfig(
            anthropic_api_key=self.anthropic_api_key,
            max_concurrent_requests=self.max_concurrent_requests
        )


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def init_config(config_file: Optional[str] = None) -> Config:
    """Initialize global configuration"""
    global _config
    _config = Config()
    return _config





# =============================================================================
# LOGGING UTILITIES - Merged from utils.py
# =============================================================================

def configure_langchain_logging(log_level: str = "WARNING") -> None:
    """
    Configure LangChain library logging levels to reduce verbosity.
    
    Args:
        log_level: Logging level for LangChain modules (default: WARNING)
    """
    langchain_modules = [
        "langchain",
        "langchain_core", 
        "langchain_community",
        "langchain_huggingface"
    ]
    
    level = getattr(logging, log_level.upper())
    for module in langchain_modules:
        logging.getLogger(module).setLevel(level)


def setup_logging(
    name: str = "dd_checklist", 
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up standard Python logging with rotating file handler
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate setup if logger already has handlers
    if logger.handlers:
        return logger
    
    # Use configured log level if not provided
    if log_level is None:
        try:
            config = get_config()
            log_level = config.log_level
        except Exception:
            log_level = "INFO"  # fallback
        
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler (if possible)
    if log_file or True:  # Always try to set up file logging
        try:
            log_dir = Path(".logs")
            log_dir.mkdir(exist_ok=True)
            
            if not log_file:
                log_file = log_dir / f"dd_checklist_{datetime.now().strftime('%Y%m%d')}.log"
            
            # Use RotatingFileHandler for better log management
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception:
            # File logging not available (e.g., on Streamlit Cloud)
            pass
    
    return logger


# Global logger instance
logger = setup_logging()


# =============================================================================
# STREAMLIT UTILITIES - Merged from utils.py
# =============================================================================

def show_success(message: str):
    """Show success message in Streamlit"""
    if STREAMLIT_AVAILABLE and st:
        st.success(message)
    logger.info(message)


def show_info(message: str):
    """Show info message in Streamlit"""
    if STREAMLIT_AVAILABLE and st:
        st.info(message)
    logger.info(message)


def show_error(message: str):
    """Show error message in Streamlit"""
    if STREAMLIT_AVAILABLE and st:
        st.error(message)
    logger.error(message)


# =============================================================================
# FILE UTILITIES - Common patterns extracted for reuse
# =============================================================================

def get_mime_type(file_path: Path) -> str:
    """Get MIME type based on file extension"""
    file_extension = file_path.suffix.lower()
    if file_extension == '.pdf':
        return 'application/pdf'
    elif file_extension in ['.doc', '.docx']:
        return 'application/msword'
    elif file_extension == '.txt':
        return 'text/plain'
    elif file_extension == '.md':
        return 'text/markdown'
    else:
        return 'application/octet-stream'


def format_document_title(doc_name: str) -> str:
    """Format document name into a readable title"""
    if '.' in doc_name:
        doc_title = doc_name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
    else:
        doc_title = doc_name.replace('_', ' ').replace('-', ' ').title()
    return doc_title


def count_documents_in_directory(directory: Path, supported_extensions: Optional[List[str]] = None) -> int:
    """Count supported documents in a directory recursively"""
    if supported_extensions is None:
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
    
    return sum(1 for f in directory.rglob('*') 
               if f.is_file() and f.suffix.lower() in supported_extensions)