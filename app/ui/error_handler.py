#!/usr/bin/env python3
"""
Standardized Error Handling System

Provides consistent error handling patterns across all modules.
Centralizes error logging, user messaging, and recovery mechanisms.
"""

import logging
import streamlit as st
from typing import Any, Optional, Callable, TypeVar
from functools import wraps

from app.core.exceptions import (
    AppException, ValidationError, ProcessingError,
    AIError, ConfigError
)

logger = logging.getLogger(__name__)

# Re-export core exceptions for backward compatibility
AppError = AppException

T = TypeVar('T')


# Exception classes are imported from app.core.exceptions above


class ErrorHandler:
    """
    Centralized error handling system with consistent patterns.
    """

    @staticmethod
    def handle_error(
        error: Exception,
        context: str = "",
        show_user_message: bool = True,
        log_error: bool = True,
        recovery_hint: Optional[str] = None
    ) -> None:
        """
        Handle an error with consistent logging and user messaging.

        Args:
            error: The exception that occurred
            context: Description of where the error occurred
            show_user_message: Whether to show error message to user
            log_error: Whether to log the error
            recovery_hint: Optional hint for user recovery
        """
        if log_error:
            ErrorHandler._log_error(error, context)

        if show_user_message:
            ErrorHandler._show_user_error(error, recovery_hint)

    @staticmethod
    def _log_error(error: Exception, context: str = "") -> None:
        """Log error with appropriate level based on error type"""
        error_msg = f"{context}: {str(error)}" if context else str(error)

        if isinstance(error, (ValidationError, ConfigError)):
            logger.warning(error_msg)
        elif isinstance(error, (ProcessingError, AIError)):
            logger.error(error_msg)
        else:
            logger.exception(f"Unexpected error - {error_msg}")

    @staticmethod
    def _show_user_error(error: Exception, recovery_hint: Optional[str] = None) -> None:
        """Show appropriate error message to user"""
        from app.ui.ui_components import status_message

        if isinstance(error, AppError):
            user_message = error.user_message
        else:
            # For unexpected errors, don't show internal details
            user_message = "An unexpected error occurred. Please try again."

        # Add recovery hint if provided
        if recovery_hint:
            user_message += f"\n\n💡 {recovery_hint}"

        # Show error message to user
        if isinstance(error, ValidationError):
            status_message(user_message, "warning")
        else:
            status_message(user_message, "error")

    @staticmethod
    def handle_with_recovery(
        func: Callable[..., T],
        context: str = "",
        default_value: Any = None,
        show_spinner: bool = False,
        spinner_text: str = "Processing...",
        recovery_hint: Optional[str] = None
    ) -> Callable[..., T]:
        """
        Decorator that provides consistent error handling with recovery.

        Args:
            func: Function to wrap
            context: Description of the operation
            default_value: Value to return on error
            show_spinner: Whether to show spinner during operation
            spinner_text: Text to show in spinner
            recovery_hint: Hint for user recovery

        Returns:
            Wrapped function with error handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                if show_spinner:
                    with st.spinner(spinner_text):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(e, context, recovery_hint=recovery_hint)
                return default_value

        return wrapper

    @staticmethod
    def validate_input(value: Any, validator: Callable[[Any], bool], error_message: str) -> bool:
        """
        Validate input with consistent error handling.

        Args:
            value: Value to validate
            validator: Function that returns True if valid
            error_message: Error message if validation fails

        Returns:
            True if valid, False otherwise
        """
        try:
            if validator(value):
                return True
            else:
                raise ValidationError(error_message)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}")

    @staticmethod
    def ensure_config_value(config_value: Any, config_name: str) -> Any:
        """
        Ensure a configuration value exists and is valid.

        Args:
            config_value: The configuration value to check
            config_name: Name of the configuration for error messages

        Returns:
            The config value if valid

        Raises:
            ConfigError: If config value is missing or invalid
        """
        if config_value is None or config_value == "":
            raise ConfigError(
                f"Configuration '{config_name}' is missing or empty",
                user_message=f"Configuration error: {config_name} is not set",
                recovery_hint="Please check your configuration and environment variables"
            )
        return config_value

    @staticmethod
    def handle_file_operation(
        file_path: str,
        operation: Callable[[], T],
        operation_name: str = "file operation"
    ) -> T:
        """
        Handle file operations with consistent error handling.

        Args:
            file_path: Path to the file being operated on
            operation: Function that performs the file operation
            operation_name: Description of the operation

        Returns:
            Result of the file operation
        """
        try:
            return operation()
        except FileNotFoundError:
            raise ProcessingError(
                f"File not found: {file_path}",
                user_message=f"File not found: {file_path}",
                recovery_hint="Please ensure the file exists and try again"
            )
        except PermissionError:
            raise ProcessingError(
                f"Permission denied accessing file: {file_path}",
                user_message=f"Cannot access file: {file_path}",
                recovery_hint="Please check file permissions"
            )
        except Exception as e:
            raise ProcessingError(
                f"Failed to {operation_name} file {file_path}: {str(e)}",
                user_message=f"File operation failed: {operation_name}",
                recovery_hint="Please check the file and try again"
            )


# Convenience decorators for common patterns
def handle_ui_errors(context: str = "", recovery_hint: Optional[str] = None):
    """
    Decorator for UI operations that need error handling.

    Args:
        context: Description of the operation
        recovery_hint: Optional hint for user recovery
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(e, context, recovery_hint=recovery_hint)
                return None
        return wrapper
    return decorator


def handle_processing_errors(context: str = "", recovery_hint: Optional[str] = None):
    """
    Decorator for processing operations that need error handling.

    Args:
        context: Description of the operation
        recovery_hint: Optional hint for user recovery
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(e, context, recovery_hint=recovery_hint)
                raise  # Re-raise for caller to handle
        return wrapper
    return decorator


def validate_and_execute(
    validator: Callable[[], bool],
    operation: Callable[[], T],
    validation_error_msg: str = "Validation failed",
    context: str = ""
) -> T:
    """
    Validate and execute operation with consistent error handling.

    Args:
        validator: Function that returns True if validation passes
        operation: Function to execute if validation passes
        validation_error_msg: Error message for validation failure
        context: Description of the operation

    Returns:
        Result of the operation

    Raises:
        ValidationError: If validation fails
    """
    try:
        if not validator():
            raise ValidationError(validation_error_msg, recovery_hint="Please check your input and try again")
        return operation()
    except ValidationError:
        raise
    except Exception as e:
        ErrorHandler.handle_error(e, f"{context} - validation/execution failed")
        raise
