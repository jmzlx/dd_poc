#!/usr/bin/env python3
"""
Test script to verify improved error handling in AI client.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.ai_client import AIClient
from app.services.ai_config import AIConfig
from app.core import LLMAuthenticationError, LLMTimeoutError, LLMQuotaExceededError, LLMConnectionError, LLMInvalidResponseError
from anthropic import AuthenticationError, APITimeoutError, RateLimitError, APIConnectionError, BadRequestError

def test_error_classification():
    """Test that error classification works correctly."""
    print("Testing improved error classification...")

    # Create a dummy config (won't actually connect)
    config = AIConfig(api_key="dummy", model="claude-3-sonnet-20240229")

    # Create AI client instance
    client = AIClient(config)

    # Test string-based fallback classification (still important for robustness)
    class MockAuthError(Exception):
        def __init__(self, message):
            super().__init__(message)

    class MockTimeoutError(Exception):
        def __init__(self, message):
            super().__init__(message)

    # Test authentication error classification (string fallback)
    try:
        auth_error = MockAuthError("Authentication failed: Invalid API key")
        client._handle_llm_error(auth_error)
        print("❌ Authentication error not caught properly")
    except LLMAuthenticationError:
        print("✅ Authentication error (string fallback) classified correctly")

    # Test timeout error classification (string fallback)
    try:
        timeout_error = MockTimeoutError("Request timed out after 30 seconds")
        client._handle_llm_error(timeout_error)
        print("❌ Timeout error not caught properly")
    except LLMTimeoutError:
        print("✅ Timeout error (string fallback) classified correctly")

    # Test quota error classification (string fallback)
    try:
        quota_error = MockTimeoutError("Rate limit exceeded - please try again later")
        client._handle_llm_error(quota_error)
        print("❌ Quota error not caught properly")
    except LLMQuotaExceededError:
        print("✅ Quota error (string fallback) classified correctly")

    # Test connection error classification (string fallback)
    try:
        conn_error = MockTimeoutError("Connection failed: Network is unreachable")
        client._handle_llm_error(conn_error)
        print("❌ Connection error not caught properly")
    except LLMConnectionError:
        print("✅ Connection error (string fallback) classified correctly")

    # Test invalid response error classification (string fallback)
    try:
        invalid_error = MockTimeoutError("Invalid response received from API")
        client._handle_llm_error(invalid_error, include_invalid_response=True)
        print("❌ Invalid response error not caught properly")
    except LLMInvalidResponseError:
        print("✅ Invalid response error (string fallback) classified correctly")

    # Test that imports work correctly
    try:
        from anthropic import AuthenticationError, APITimeoutError, RateLimitError
        print("✅ Anthropic exception imports working correctly")
    except ImportError as e:
        print(f"⚠️  Anthropic imports failed (expected in some environments): {e}")

    print("\nAll error classification tests completed!")

if __name__ == "__main__":
    test_error_classification()
