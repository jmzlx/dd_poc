"""
Core Business Logic Layer

This layer contains the core business logic and domain models.
It should not depend on UI or external frameworks.
"""

# Configuration
from .config import AppConfig, get_config

# Exceptions
from .exceptions import (
    AppException,
    DocumentProcessingError,
    SearchError,
    ConfigError,
    FileOperationError,
    AIError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMTimeoutError,
    LLMQuotaExceededError,
    LLMInvalidResponseError,
    create_processing_error,
    create_config_error,
    create_ai_error
)

# Core classes and functions
from .document_processor import DocumentProcessor
from .search import search_and_analyze, search_documents
from .ranking import rerank_results
from .parsers import parse_checklist, parse_questions
from .utils import create_document_processor, format_document_title, count_documents_in_directory
from .logging import logger
from .constants import (
    RELEVANCY_THRESHOLD,
    SIMILARITY_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    QA_MAX_TOKENS,
    CHECKLIST_PARSING_MAX_TOKENS
)

__all__ = [
    # Configuration
    'AppConfig', 'get_config',

    # Exceptions
    'AppException', 'DocumentProcessingError', 'SearchError', 'ConfigError',
    'FileOperationError', 'AIError', 'LLMConnectionError', 'LLMAuthenticationError',
    'LLMTimeoutError', 'LLMQuotaExceededError', 'LLMInvalidResponseError',
    'create_processing_error', 'create_config_error', 'create_ai_error',

    # Core functionality
    'DocumentProcessor', 'search_and_analyze', 'search_documents', 'rerank_results',
    'parse_checklist', 'parse_questions', 'create_document_processor',
    'format_document_title', 'count_documents_in_directory', 'logger',

    # Constants
    'RELEVANCY_THRESHOLD', 'SIMILARITY_THRESHOLD', 'DEFAULT_BATCH_SIZE', 'QA_MAX_TOKENS', 'CHECKLIST_PARSING_MAX_TOKENS'
]
