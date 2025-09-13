#!/usr/bin/env python3
"""
AI Configuration

Configuration settings for AI service operations.
Provides type safety and validation for AI service parameters.
"""

from dataclasses import dataclass

from app.core.exceptions import ConfigError
from app.core.constants import TEMPERATURE


@dataclass
class AIConfig:
    """
    Configuration settings for AI service operations.

    This dataclass encapsulates all configuration parameters needed
    for AI service initialization and operation, providing type safety
    and validation.

    Attributes:
        api_key: Anthropic API key for authentication
        model: Claude model name to use for operations
        temperature: Sampling temperature (0.0 = deterministic, higher = more creative)
        max_tokens: Maximum tokens to generate in responses

    Example:
        config = AIConfig(
            api_key="sk-ant-...",
            model="claude-3-5-sonnet",
            temperature=TEMPERATURE,
            max_tokens=4000
        )
    """
    api_key: str
    model: str
    temperature: float = TEMPERATURE
    max_tokens: int = 4000

    def validate(self) -> None:
        """
        Validate the AI configuration for required values and consistency.

        Performs comprehensive validation of all configuration parameters
        to ensure they are valid for AI service operations.

        Raises:
            ConfigError: If any configuration values are invalid
        """
        if not self.api_key or not self.api_key.strip():
            raise ConfigError(
                "AI API key is missing",
                user_message="API key is required for AI features",
                recovery_hint="Please configure your Anthropic API key in the sidebar"
            )

        if not self.model or not self.model.strip():
            raise ConfigError(
                "AI model is not specified",
                user_message="AI model selection is required",
                recovery_hint="Please select a Claude model"
            )
