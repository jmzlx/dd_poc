from typing import Dict, Any, Optional
from pathlib import Path
import os
from dotenv import load_dotenv
from app.core.constants import (
    CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_THRESHOLD,
    RELEVANCY_THRESHOLD, CLASSIFICATION_MAX_TOKENS, CHECKLIST_PARSING_MAX_TOKENS,
    TEMPERATURE, STATISTICAL_CANDIDATE_POOL_SIZE, STATISTICAL_STD_MULTIPLIER,
    STATISTICAL_MIN_CANDIDATES, STATISTICAL_MIN_STD_DEV
)

load_dotenv()


class AppConfig:
    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        self._config['ui'] = {
            'page_title': "🤖 AI Due Diligence",
            'page_icon': "🤖",
            'layout': "wide",
            'top_k_search_results': 10
        }

        self._config['model'] = {
            'sentence_transformer_model': 'sentence-transformers/all-mpnet-base-v2',
            'claude_model': os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514'),
            'claude_haiku_model': 'claude-3-5-haiku-20241022',
            'classification_max_tokens': CLASSIFICATION_MAX_TOKENS,
            'temperature': float(os.getenv('CLAUDE_TEMPERATURE', str(TEMPERATURE))),
            'max_tokens': int(os.getenv('CLAUDE_MAX_TOKENS', '16000'))  # High limit for checklist parsing
        }

        self._config['processing'] = {
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'relevancy_threshold': RELEVANCY_THRESHOLD,
            'supported_file_extensions': [
                '.pdf', '.docx', '.doc', '.txt', '.md',
                '.xls', '.xlsx', '.ppt', '.pptx'
            ],
            'faiss_store_name': 'default',
            # Statistical filtering configuration
            'statistical_candidate_pool_size': STATISTICAL_CANDIDATE_POOL_SIZE,
            'statistical_std_multiplier': STATISTICAL_STD_MULTIPLIER,
            'statistical_min_candidates': STATISTICAL_MIN_CANDIDATES,
            'statistical_min_std_dev': STATISTICAL_MIN_STD_DEV
        }

        # Use absolute paths to avoid working directory issues
        project_root = Path(__file__).parent.parent.parent  # Go up from app/core/config.py to project root
        
        self._config['paths'] = {
            'data_dir': project_root / 'data',
            'strategy_dir': project_root / 'data/strategy',
            'checklist_dir': project_root / 'data/checklist',
            'questions_dir': project_root / 'data/questions',
            'vdrs_dir': project_root / 'data/vdrs',
            'faiss_dir': project_root / 'data/search_indexes'
        }

        self._config['anthropic'] = {
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'model': os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet')
        }

    @property
    def ui(self) -> Dict[str, Any]:
        return self._config['ui']

    @property
    def model(self) -> Dict[str, Any]:
        return self._config['model']

    @property
    def processing(self) -> Dict[str, Any]:
        return self._config['processing']

    @property
    def paths(self) -> Dict[str, Path]:
        return self._config['paths']

    @property
    def anthropic(self) -> Dict[str, Optional[str]]:
        return self._config['anthropic']

    def validate(self) -> bool:
        """Validate all critical configuration values."""
        self._validate_anthropic_config()
        self._validate_paths()
        self._validate_models()
        self._validate_processing_config()
        self._validate_file_extensions()
        return True

    def _validate_anthropic_config(self) -> None:
        """Validate Anthropic API configuration."""
        if not self.anthropic.get('api_key'):
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        model = self.anthropic.get('model')
        if not model:
            raise ValueError("CLAUDE_MODEL environment variable is required")

        valid_claude_models = [
            'claude-sonnet-4-20250514',
            'claude-opus-4-1-20250805',
            'claude-3-5-haiku-20241022'
        ]
        if model not in valid_claude_models:
            raise ValueError(f"Invalid Claude model: {model}. Valid models: {', '.join(valid_claude_models)}")

    def _validate_paths(self) -> None:
        """Validate that critical directories exist."""
        critical_dirs = [
            ('data_dir', self.paths['data_dir']),
            ('vdrs_dir', self.paths['vdrs_dir'])
        ]

        for dir_name, dir_path in critical_dirs:
            if not dir_path.exists():
                raise ValueError(f"Critical directory '{dir_name}' does not exist: {dir_path}")
            if not dir_path.is_dir():
                raise ValueError(f"Path '{dir_name}' exists but is not a directory: {dir_path}")

    def _validate_models(self) -> None:
        """Validate that required models are available."""
        # Check sentence transformer model
        model_path = Path('models') / 'sentence_transformers' / self.model['sentence_transformer_model'].split('/')[-1]
        if not model_path.exists():
            raise ValueError(f"Sentence transformer model not found: {model_path}")

        # Check cross-encoder model
        cross_encoder_path = Path('models') / 'cross_encoder' / 'ms-marco-MiniLM-L-6-v2'
        if not cross_encoder_path.exists():
            raise ValueError(f"Cross-encoder model not found: {cross_encoder_path}")

    def _validate_processing_config(self) -> None:
        """Validate processing configuration values."""
        processing = self.processing

        # Validate chunk size
        chunk_size = processing['chunk_size']
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {chunk_size}. Must be a positive integer.")

        # Validate chunk overlap
        chunk_overlap = processing['chunk_overlap']
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError(f"Invalid chunk_overlap: {chunk_overlap}. Must be a non-negative integer.")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

        # Validate thresholds
        similarity_threshold = processing['similarity_threshold']
        if not isinstance(similarity_threshold, (int, float)) or not (0 <= similarity_threshold <= 1):
            raise ValueError(f"Invalid similarity_threshold: {similarity_threshold}. Must be between 0 and 1.")

        relevancy_threshold = processing['relevancy_threshold']
        if not isinstance(relevancy_threshold, (int, float)) or not (0 <= relevancy_threshold <= 1):
            raise ValueError(f"Invalid relevancy_threshold: {relevancy_threshold}. Must be between 0 and 1.")

        # Validate max tokens
        max_tokens = processing.get('classification_max_tokens', CLASSIFICATION_MAX_TOKENS)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"Invalid classification_max_tokens: {max_tokens}. Must be a positive integer.")

    def _validate_file_extensions(self) -> None:
        """Validate supported file extensions."""
        extensions = self.processing['supported_file_extensions']
        if not extensions:
            raise ValueError("supported_file_extensions cannot be empty")

        # Validate each extension starts with a dot and contains valid characters
        for ext in extensions:
            if not isinstance(ext, str):
                raise ValueError(f"Invalid file extension type: {type(ext)}. Must be string.")
            if not ext.startswith('.'):
                raise ValueError(f"File extension must start with '.': {ext}")
            if len(ext) < 2 or not ext[1:].replace('_', '').replace('-', '').isalnum():
                raise ValueError(f"Invalid file extension format: {ext}")

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions for document processing."""
        return self._config['processing']['supported_file_extensions']


# Global configuration instance
_config_instance: Optional[AppConfig] = None


def get_app_config() -> AppConfig:
    """Get the global application configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
        _config_instance.validate()
    return _config_instance


# Compatibility alias
init_app_config = get_app_config

# Compatibility alias
get_config = get_app_config
