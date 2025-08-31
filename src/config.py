#!/usr/bin/env python3
"""
Configuration Management Module

This module centralizes all configuration settings for the DD-Checklist application.
Handles environment variables, default settings, and configuration validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Fix tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    claude_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 2000
    embedding_dimension: int = 384
    
    def __post_init__(self):
        """Load model configuration from environment variables"""
        self.sentence_transformer_model = os.getenv('SENTENCE_TRANSFORMER_MODEL', self.sentence_transformer_model)
        self.claude_model = os.getenv('CLAUDE_MODEL', self.claude_model)
        self.temperature = float(os.getenv('CLAUDE_TEMPERATURE', str(self.temperature)))
        self.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', str(self.max_tokens)))
        self.embedding_dimension = int(os.getenv('EMBEDDING_DIMENSION', str(self.embedding_dimension)))


@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    chunk_size: int = 400
    chunk_overlap: int = 50
    max_text_length: int = 10000
    batch_size: int = 100
    description_batch_size: int = 100
    similarity_threshold: float = 0.35
    relevancy_threshold: float = 0.4
    primary_threshold: float = 0.5
    min_display_threshold: float = 0.15
    max_workers: int = 4
    file_timeout: int = 30
    skip_descriptions: bool = False
    supported_file_extensions: List[str] = field(
        default_factory=lambda: ['.pdf', '.docx', '.doc', '.txt', '.md']
    )
    
    def __post_init__(self):
        """Load processing configuration from environment variables"""
        self.chunk_size = int(os.getenv('CHUNK_SIZE', str(self.chunk_size)))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', str(self.chunk_overlap)))
        self.max_text_length = int(os.getenv('MAX_TEXT_LENGTH', str(self.max_text_length)))
        self.batch_size = int(os.getenv('BATCH_SIZE', str(self.batch_size)))
        self.description_batch_size = int(os.getenv('DESCRIPTION_BATCH_SIZE', str(self.description_batch_size)))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', str(self.similarity_threshold)))
        self.relevancy_threshold = float(os.getenv('RELEVANCY_THRESHOLD', str(self.relevancy_threshold)))
        self.primary_threshold = float(os.getenv('PRIMARY_THRESHOLD', str(self.primary_threshold)))
        self.min_display_threshold = float(os.getenv('MIN_DISPLAY_THRESHOLD', str(self.min_display_threshold)))
        self.max_workers = int(os.getenv('MAX_WORKERS', str(self.max_workers)))
        self.file_timeout = int(os.getenv('FILE_TIMEOUT', str(self.file_timeout)))
        self.skip_descriptions = os.getenv('SKIP_DESCRIPTIONS', 'false').lower() == 'true'
        
        # Handle file extensions from environment (comma-separated)
        extensions_env = os.getenv('SUPPORTED_FILE_EXTENSIONS')
        if extensions_env:
            self.supported_file_extensions = [ext.strip() for ext in extensions_env.split(',')]


@dataclass
class UIConfig:
    """Configuration for UI settings"""
    page_title: str = "AI Due Diligence"
    page_icon: str = "🤖"
    layout: str = "wide"
    top_k_search_results: int = 5
    max_question_sources: int = 3
    max_checklist_matches: int = 5


@dataclass
class PathConfig:
    """Configuration for file paths"""
    data_dir: str = "data"
    checklist_dir: str = "data/checklist"
    questions_dir: str = "data/questions"
    strategy_dir: str = "data/strategy"
    vdrs_dir: str = "data/vdrs"
    cache_dir: str = ".cache"
    
    def __post_init__(self):
        """Convert string paths to Path objects and ensure they exist"""
        self.data_path = Path(self.data_dir)
        self.checklist_path = Path(self.checklist_dir)
        self.questions_path = Path(self.questions_dir)
        self.strategy_path = Path(self.strategy_dir)
        self.vdrs_path = Path(self.vdrs_dir)
        self.cache_path = Path(self.cache_dir)


@dataclass
class APIConfig:
    """Configuration for API settings"""
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    max_concurrent_requests: int = 50
    request_timeout: int = 30
    retry_attempts: int = 3
    base_delay: float = 0.2
    max_retries: int = 2
    batch_retry_attempts: int = 1
    batch_base_delay: float = 0.1
    single_retry_base_delay: float = 0.05
    
    def __post_init__(self):
        """Load API configuration from environment variables"""
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.openai_api_key:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        self.max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', str(self.max_concurrent_requests)))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', str(self.request_timeout)))
        self.retry_attempts = int(os.getenv('RETRY_ATTEMPTS', str(self.retry_attempts)))
        self.base_delay = float(os.getenv('BASE_DELAY', str(self.base_delay)))
        self.max_retries = int(os.getenv('MAX_RETRIES', str(self.max_retries)))
        self.batch_retry_attempts = int(os.getenv('BATCH_RETRY_ATTEMPTS', str(self.batch_retry_attempts)))
        self.batch_base_delay = float(os.getenv('BATCH_BASE_DELAY', str(self.batch_base_delay)))
        self.single_retry_base_delay = float(os.getenv('SINGLE_RETRY_BASE_DELAY', str(self.single_retry_base_delay)))


@dataclass
class AppConfig:
    """Main application configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Environment settings
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Load environment-specific settings"""
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')


class ConfigManager:
    """
    Configuration manager that handles loading and validating configuration
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional path to configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize configuration
        self.config = AppConfig()
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_file(self, config_file: str) -> None:
        """
        Load configuration from file (JSON or YAML)
        
        Args:
            config_file: Path to configuration file
        """
        import json
        
        config_path = Path(config_file)
        
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self._update_config_from_dict(config_data)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    self._update_config_from_dict(config_data)
                except ImportError:
                    print("PyYAML not installed. Cannot load YAML configuration.")
        except Exception as e:
            print(f"Warning: Could not load configuration from {config_file}: {e}")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary
        
        Args:
            config_data: Configuration dictionary
        """
        for section, values in config_data.items():
            if hasattr(self.config, section) and isinstance(values, dict):
                config_section = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(config_section, key):
                        setattr(config_section, key, value)
    
    def _validate_config(self) -> None:
        """Validate configuration settings"""
        # Validate paths
        if not self.config.paths.data_path.exists():
            print(f"Warning: Data directory does not exist: {self.config.paths.data_path}")
        
        # Validate model settings
        if self.config.processing.chunk_size <= self.config.processing.chunk_overlap:
            print("Warning: Chunk size should be larger than chunk overlap")
        
        # Validate thresholds
        if not 0 <= self.config.processing.similarity_threshold <= 1:
            print("Warning: Similarity threshold should be between 0 and 1")
    
    def get_config(self) -> AppConfig:
        """Get the current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration settings
        
        Args:
            **kwargs: Configuration updates
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def update_processing_config(self, **kwargs) -> None:
        """
        Update processing configuration dynamically
        
        Args:
            **kwargs: Processing configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config.processing, key):
                setattr(self.config.processing, key, value)
            else:
                print(f"Warning: Unknown processing config key: {key}")
    
    def update_api_config(self, **kwargs) -> None:
        """
        Update API configuration dynamically
        
        Args:
            **kwargs: API configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config.api, key, value)
            else:
                print(f"Warning: Unknown API config key: {key}")
    
    def save_config(self, config_file: str) -> None:
        """
        Save current configuration to file
        
        Args:
            config_file: Path to save configuration
        """
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self.config)
        
        # Remove Path objects and other non-serializable items
        config_dict = self._make_serializable(config_dict)
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make configuration dictionary serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() 
                   if not k.endswith('_path')}  # Skip Path objects
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance
    
    Returns:
        Application configuration
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_config()


def init_config(config_file: Optional[str] = None) -> ConfigManager:
    """
    Initialize global configuration
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        Configuration manager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_file)
    return _config_manager


def update_config(**kwargs) -> None:
    """
    Update global configuration
    
    Args:
        **kwargs: Configuration updates
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    _config_manager.update_config(**kwargs)


# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "processing": {
        "batch_size": 50,
        "similarity_threshold": 0.3
    },
    "ui": {
        "layout": "wide"
    }
}

PRODUCTION_CONFIG = {
    "processing": {
        "batch_size": 100,
        "similarity_threshold": 0.35
    },
    "api": {
        "max_concurrent_requests": 20,
        "request_timeout": 60
    }
}

STREAMLIT_CLOUD_CONFIG = {
    "processing": {
        "batch_size": 100,  # Optimized for performance
        "description_batch_size": 100,  # Match summary batch size
        "max_text_length": 8000,  # Higher limit for better quality
        "max_workers": 2,  # Moderate parallelism for cloud
        "file_timeout": 30  # Standard timeout
    },
    "api": {
        "max_concurrent_requests": 30,  # Good concurrency for cloud
        "base_delay": 0.1,  # Fast delays
        "batch_base_delay": 0.05,  # Very fast batches
        "request_timeout": 30
    }
}


def get_environment_config() -> Dict[str, Any]:
    """
    Get environment-specific configuration
    
    Returns:
        Environment configuration dictionary
    """
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return PRODUCTION_CONFIG
    elif env == 'streamlit_cloud':
        return STREAMLIT_CLOUD_CONFIG
    else:
        return DEVELOPMENT_CONFIG


# Utility functions for common configuration access
def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return get_config().model


def get_processing_config() -> ProcessingConfig:
    """Get processing configuration"""
    return get_config().processing


def get_ui_config() -> UIConfig:
    """Get UI configuration"""
    return get_config().ui


def get_path_config() -> PathConfig:
    """Get path configuration"""
    return get_config().paths


def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_config().api


def is_ai_enabled() -> bool:
    """Check if AI features are enabled (API key available)"""
    api_config = get_api_config()
    return api_config.anthropic_api_key is not None


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions"""
    return get_processing_config().supported_file_extensions


def update_processing_config(**kwargs) -> None:
    """Update processing configuration dynamically"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    _config_manager.update_processing_config(**kwargs)


def update_api_config(**kwargs) -> None:
    """Update API configuration dynamically"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    _config_manager.update_api_config(**kwargs)
