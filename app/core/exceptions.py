#!/usr/bin/env python3
"""
Core Exception Classes

Centralized exception definitions for the application.
This module provides clean exception classes without
depending on UI or external frameworks.
"""


class AppException(Exception):
    """Base exception class for application-specific errors"""

    def __init__(self, message: str, user_message: str = None, recovery_hint: str = None):
        self.message = message
        self.user_message = user_message or message
        self.recovery_hint = recovery_hint
        super().__init__(message)


class ValidationError(AppException):
    """Error for input validation failures"""
    pass


class ProcessingError(AppException):
    """Error for document processing failures"""
    pass


class AIError(AppException):
    """Error for AI service failures"""
    pass


class ConfigError(AppException):
    """Error for configuration issues"""
    pass


class FileOperationError(AppException):
    """Error for file operation failures"""
    pass


class NetworkError(AppException):
    """Error for network-related failures"""
    pass


class LLMConnectionError(AIError):
    """Error for LLM API connection failures"""
    pass


class LLMAuthenticationError(AIError):
    """Error for LLM API authentication failures"""
    pass


class LLMTimeoutError(AIError):
    """Error for LLM API timeout failures"""
    pass


class LLMQuotaExceededError(AIError):
    """Error for LLM API quota/rate limit exceeded"""
    pass


class LLMInvalidResponseError(AIError):
    """Error for invalid LLM API responses"""
    pass


class DocumentProcessingError(ProcessingError):
    """Error for document processing failures"""
    pass


class SearchError(AppException):
    """Error for search operation failures"""
    pass


# Convenience functions for creating exceptions
def create_validation_error(message: str, recovery_hint: str = None) -> ValidationError:
    """Create a validation error with consistent formatting"""
    return ValidationError(
        message,
        user_message=f"Validation error: {message}",
        recovery_hint=recovery_hint or "Please check your input and try again"
    )


def create_processing_error(message: str, recovery_hint: str = None) -> ProcessingError:
    """Create a processing error with consistent formatting"""
    return ProcessingError(
        message,
        user_message=f"Processing error: {message}",
        recovery_hint=recovery_hint or "Please check your files and try again"
    )


def create_ai_error(message: str, recovery_hint: str = None) -> AIError:
    """Create an AI error with consistent formatting"""
    return AIError(
        message,
        user_message=f"AI service error: {message}",
        recovery_hint=recovery_hint or "Please check your API key and try again"
    )


def create_config_error(message: str, recovery_hint: str = None) -> ConfigError:
    """Create a configuration error with consistent formatting"""
    return ConfigError(
        message,
        user_message=f"Configuration error: {message}",
        recovery_hint=recovery_hint or "Please check your configuration and environment variables"
    )


def create_file_error(message: str, recovery_hint: str = None) -> FileOperationError:
    """Create a file operation error with consistent formatting"""
    return FileOperationError(
        message,
        user_message=f"File error: {message}",
        recovery_hint=recovery_hint or "Please check file permissions and paths"
    )


def create_network_error(message: str, recovery_hint: str = None) -> NetworkError:
    """Create a network error with consistent formatting"""
    return NetworkError(
        message,
        user_message=f"Network error: {message}",
        recovery_hint=recovery_hint or "Please check your internet connection and try again"
    )


def create_llm_connection_error(message: str, recovery_hint: str = None) -> LLMConnectionError:
    """Create an LLM connection error with consistent formatting"""
    return LLMConnectionError(
        message,
        user_message=f"AI service connection error: {message}",
        recovery_hint=recovery_hint or "Please check your internet connection and try again"
    )


def create_llm_authentication_error(message: str, recovery_hint: str = None) -> LLMAuthenticationError:
    """Create an LLM authentication error with consistent formatting"""
    return LLMAuthenticationError(
        message,
        user_message=f"AI service authentication error: {message}",
        recovery_hint=recovery_hint or "Please check your API key and try again"
    )


def create_llm_timeout_error(message: str, recovery_hint: str = None) -> LLMTimeoutError:
    """Create an LLM timeout error with consistent formatting"""
    return LLMTimeoutError(
        message,
        user_message=f"AI service timeout: {message}",
        recovery_hint=recovery_hint or "Please try again in a few moments"
    )


def create_llm_quota_error(message: str, recovery_hint: str = None) -> LLMQuotaExceededError:
    """Create an LLM quota exceeded error with consistent formatting"""
    return LLMQuotaExceededError(
        message,
        user_message=f"AI service quota exceeded: {message}",
        recovery_hint=recovery_hint or "Please check your API usage limits and try again later"
    )


def create_llm_invalid_response_error(message: str, recovery_hint: str = None) -> LLMInvalidResponseError:
    """Create an LLM invalid response error with consistent formatting"""
    return LLMInvalidResponseError(
        message,
        user_message=f"AI service returned invalid response: {message}",
        recovery_hint=recovery_hint or "Please try again or contact support if the issue persists"
    )


def create_document_processing_error(message: str, recovery_hint: str = None) -> DocumentProcessingError:
    """Create a document processing error with consistent formatting"""
    return DocumentProcessingError(
        message,
        user_message=f"Document processing error: {message}",
        recovery_hint=recovery_hint or "Please check your document format and try again"
    )


def create_search_error(message: str, recovery_hint: str = None) -> SearchError:
    """Create a search error with consistent formatting"""
    return SearchError(
        message,
        user_message=f"Search error: {message}",
        recovery_hint=recovery_hint or "Please try adjusting your search terms"
    )
