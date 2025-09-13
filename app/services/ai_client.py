#!/usr/bin/env python3
"""
AI Client

Handles Anthropic API client and LLM interaction logic.
Provides clean interface for LLM operations and connection management.
"""

from typing import Optional, Any, List

from app.core.exceptions import AIError
from app.services.ai_config import AIConfig
from app.core.exceptions import LLMConnectionError, LLMAuthenticationError, LLMTimeoutError, LLMQuotaExceededError, LLMInvalidResponseError

# Import specific exception types for robust error handling
try:
    from anthropic import (
        APIConnectionError, APIError, APITimeoutError, AuthenticationError,
        BadRequestError, ConflictError, InternalServerError, NotFoundError,
        PermissionDeniedError, RateLimitError, UnprocessableEntityError,
        ServiceUnavailableError
    )
except ImportError:
    # Fallback if anthropic package is not directly available
    APIConnectionError = APIError = APITimeoutError = AuthenticationError = None
    BadRequestError = ConflictError = InternalServerError = NotFoundError = None
    PermissionDeniedError = RateLimitError = UnprocessableEntityError = None
    ServiceUnavailableError = None


class AIClient:
    """
    Anthropic API client for LLM interactions.

    This class manages the connection to Anthropic's Claude models,
    handles initialization, and provides methods for LLM operations.
    """

    def __init__(self, config: AIConfig) -> None:
        """
        Initialize AI client with configuration.

        Args:
            config: AIConfig object containing service configuration

        Raises:
            AIError: If initialization fails
        """
        self.config: AIConfig = config
        self._llm: Optional[Any] = None
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """
        Ensure the AI client is properly initialized and ready for use.

        This method handles lazy initialization of the AI client, creating
        the underlying LLM connection and testing it with a simple query.

        Raises:
            AIError: If initialization fails due to configuration or connection issues
        """
        if self._initialized:
            return

        try:
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                api_key=self.config.api_key,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # Test the connection with a simple query that validates AI functionality
            from langchain_core.messages import HumanMessage
            test_response = self._llm.invoke([
                HumanMessage(content="Please respond with 'AI connection successful' if you can read this message.")
            ])
            if not test_response or not hasattr(test_response, 'content') or not test_response.content.strip():
                raise AIError("AI service test failed - no valid response received")

            # Verify the response contains expected content
            response_text = test_response.content.strip().lower()
            if "successful" not in response_text and "ai" not in response_text:
                raise AIError("AI service test failed - unexpected response format")

            self._initialized = True

        except ImportError as e:
            raise AIError(
                f"Missing required AI library: {str(e)}",
                user_message="AI libraries not installed",
                recovery_hint="Please install required dependencies"
            )
        except Exception as e:
            self._handle_llm_error(e)

    def _handle_llm_error(self, error: Exception, include_invalid_response: bool = False) -> None:
        """
        Handle LLM-related errors with robust error type detection.

        This method uses exception type checking as the primary classification method,
        with string-based fallbacks for compatibility with different library versions.

        Args:
            error: The exception that occurred
            include_invalid_response: Whether to include invalid response error handling

        Raises:
            Specific LLM error types based on exception type and content
        """
        # Primary: Check exception types for robust classification
        if self._is_authentication_error(error):
            raise LLMAuthenticationError(
                f"AI authentication failed: {str(error)}",
                user_message="AI authentication failed",
                recovery_hint="Please check your API key"
            )
        elif self._is_timeout_error(error):
            raise LLMTimeoutError(
                f"AI service timeout: {str(error)}",
                user_message="AI service timed out",
                recovery_hint="Please try again later"
            )
        elif self._is_quota_error(error):
            raise LLMQuotaExceededError(
                f"AI quota exceeded: {str(error)}",
                user_message="AI quota exceeded",
                recovery_hint="Please check your API usage limits"
            )
        elif self._is_connection_error(error):
            raise LLMConnectionError(
                f"AI connection failed: {str(error)}",
                user_message="AI connection failed",
                recovery_hint="Please check your network connection"
            )
        elif include_invalid_response and self._is_invalid_response_error(error):
            raise LLMInvalidResponseError(
                f"AI returned invalid response: {str(error)}",
                user_message="AI returned invalid response",
                recovery_hint="Please try again"
            )

        # Default error messages based on context
        if include_invalid_response:
            raise AIError(
                f"Response generation failed: {str(error)}",
                user_message="Failed to generate AI response",
                recovery_hint="Please try again or check your API key"
            )
        else:
            raise AIError(
                f"Failed to initialize AI client: {str(error)}",
                user_message="AI client initialization failed",
                recovery_hint="Please check your API key and network connection"
            )

    def _is_authentication_error(self, error: Exception) -> bool:
        """Check if error is an authentication-related error."""
        # Primary: Check exception types
        if AuthenticationError and isinstance(error, AuthenticationError):
            return True
        if PermissionDeniedError and isinstance(error, PermissionDeniedError):
            return True

        # Fallback: String-based detection for compatibility
        error_msg = str(error).lower()
        return "authentication" in error_msg or "api key" in error_msg or "unauthorized" in error_msg

    def _is_timeout_error(self, error: Exception) -> bool:
        """Check if error is a timeout-related error."""
        # Primary: Check exception types
        if APITimeoutError and isinstance(error, APITimeoutError):
            return True

        # Fallback: String-based detection
        error_msg = str(error).lower()
        return "timeout" in error_msg or "timed out" in error_msg

    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is a quota/rate limit related error."""
        # Primary: Check exception types
        if RateLimitError and isinstance(error, RateLimitError):
            return True

        # Fallback: String-based detection
        error_msg = str(error).lower()
        return "quota" in error_msg or "rate limit" in error_msg or "limit exceeded" in error_msg

    def _is_connection_error(self, error: Exception) -> bool:
        """Check if error is a connection/network related error."""
        # Primary: Check exception types
        if APIConnectionError and isinstance(error, APIConnectionError):
            return True
        if InternalServerError and isinstance(error, InternalServerError):
            return True
        if ServiceUnavailableError and isinstance(error, ServiceUnavailableError):
            return True

        # Fallback: String-based detection
        error_msg = str(error).lower()
        return ("connection" in error_msg or "network" in error_msg or
                "connection reset" in error_msg or "connection refused" in error_msg)

    def _is_invalid_response_error(self, error: Exception) -> bool:
        """Check if error is related to invalid/malformed responses."""
        # Primary: Check exception types
        if BadRequestError and isinstance(error, BadRequestError):
            return True
        if UnprocessableEntityError and isinstance(error, UnprocessableEntityError):
            return True

        # Fallback: String-based detection
        error_msg = str(error).lower()
        return ("invalid" in error_msg or "malformed" in error_msg or
                "bad request" in error_msg or "unprocessable" in error_msg)

    @property
    def is_available(self) -> bool:
        """
        Check if AI client is available and ready for operations.

        This property performs lazy initialization if needed and returns
        the availability status of the AI client.

        Returns:
            True if AI client is initialized and ready, False otherwise
        """
        try:
            self._ensure_initialized()
            return True
        except (AIError):
            return False

    @property
    def llm(self) -> Any:
        """
        Get the underlying LLM instance for direct access.

        This property provides access to the raw LangChain LLM object
        for advanced use cases that require direct interaction.

        Returns:
            LangChain LLM instance (ChatAnthropic)

        Raises:
            AIError: If LLM is not initialized
        """
        self._ensure_initialized()
        return self._llm

    def generate_response(self, messages: List[dict]) -> str:
        """
        Generate a response using the LLM.

        Args:
            messages: List of message dictionaries for the LLM

        Returns:
            Generated response content

        Raises:
            AIError: If response generation fails
        """
        self._ensure_initialized()

        try:
            response = self._llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            self._handle_llm_error(e, include_invalid_response=True)

    def generate_text(self, prompt: str, context: Optional[List[str]] = None) -> str:
        """
        Generate text using the AI client.

        Args:
            prompt: The main prompt for text generation
            context: Optional context documents

        Returns:
            Generated text response
        """
        self._ensure_initialized()

        # Prepare the full prompt
        full_prompt = prompt
        if context:
            context_str = "\n\n".join(context[:3])  # Limit context to prevent token overflow
            full_prompt = f"Context:\n{context_str}\n\n{prompt}"

        try:
            from langchain_core.messages import HumanMessage

            response = self._llm.invoke([HumanMessage(content=full_prompt)])
            return response.content.strip()

        except Exception as e:
            self._handle_llm_error(e, include_invalid_response=True)
