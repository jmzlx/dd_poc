#!/usr/bin/env python3
"""
AI Service

Provides a clean interface for AI operations.
Reduces coupling between AI components and the rest of the system.
"""

from typing import Optional, Dict, List, Any

from app.core.exceptions import AIError, ConfigError
# Removed circular import: from app.ui.error_handler import handle_processing_errors
from app.core.exceptions import create_config_error
from app.core.constants import QA_MAX_TOKENS, SUPPORTED_ANALYSIS_TYPES
from app.services.ai_config import AIConfig
from app.services.ai_client import AIClient
from app.services.response_parser import ResponseParser


class AIService:
    """
    Simplified AI service providing clean, type-safe interface for AI operations.

    This service replaces the complex DDChecklistAgent with a focused, simple interface
    that handles the core AI operations needed by the application. It provides:

    Features:
    - Type-safe AI operations with comprehensive error handling
    - Multiple analysis types (overview, strategic, checklist, questions)
    - Token usage estimation and limits
    - Configurable AI models and parameters
    - Clean separation of concerns

    Attributes:
        config: AIConfig object containing service configuration
        is_available: Property indicating if service is ready for use

    Example:
        config = AIConfig(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
        ai_service = AIService(config)

        if ai_service.is_available:
            result = ai_service.analyze_documents(docs, "overview")
            answer = ai_service.answer_question("What is the revenue?", context)
    """

    def __init__(self, config: AIConfig) -> None:
        """
        Initialize AI service with configuration and validate setup.

        Args:
            config: AIConfig object containing service configuration

        Raises:
            ConfigError: If configuration validation fails
        """
        self.config: AIConfig = config
        self.config.validate()
        self._client: Optional[AIClient] = None

    @property
    def _ensure_client(self) -> AIClient:
        """
        Ensure the AI client is properly initialized.

        Returns:
            Initialized AIClient instance

        Raises:
            AIError: If client initialization fails
        """
        if self._client is None:
            self._client = AIClient(self.config)
        return self._client

    @property
    def is_available(self) -> bool:
        """
        Check if AI service is available and ready for operations.

        This property performs lazy initialization if needed and returns
        the availability status of the AI service.

        Returns:
            True if AI service is initialized and ready, False otherwise
        """
        try:
            return self._ensure_client.is_available
        except (AIError, ConfigError):
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
        return self._ensure_client.llm

    # Removed decorator to avoid circular imports
    def generate_text(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate text using the AI service.

        Args:
            prompt: The main prompt for text generation
            context: Optional context documents
            max_length: Maximum response length

        Returns:
            Generated text response
        """
        client = self._ensure_client
        response = client.generate_text(prompt, context)
        return ResponseParser.format_response(response, max_length)

    # Removed decorator to avoid circular imports
    def analyze_documents(
        self,
        documents: Dict[str, Dict[str, Any]],
        analysis_type: str,
        strategy_text: Optional[str] = None,
        checklist_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Analyze documents using AI with different analysis types.

        This method performs comprehensive document analysis using AI, supporting
        multiple analysis types for different business use cases.

        Args:
            documents: Dictionary mapping document names to document data.
                      Each document dict should contain 'content' and other metadata.
            analysis_type: Type of analysis to perform. Supported types:
                         - "overview": Company overview and business analysis
                         - "strategic": Strategic positioning and recommendations
                         - "checklist": Due diligence checklist analysis
                         - "questions": Answer due diligence questions
            strategy_text: Optional strategy document content for context
            checklist_results: Optional existing checklist results for strategic analysis

        Returns:
            AI-generated analysis text with comprehensive insights

        Raises:
            AIError: If analysis fails or service is unavailable
            ValueError: If analysis_type is not supported

        Example:
            docs = {
                "annual_report.pdf": {"content": "Company financials...", "name": "Annual Report"},
                "strategy.docx": {"content": "Strategic plan...", "name": "Strategy"}
            }
            analysis = ai_service.analyze_documents(docs, "overview")
        """
        # Input validation
        if not documents:
            raise ValueError("Documents dictionary cannot be None or empty")

        if not isinstance(documents, dict):
            raise ValueError("Documents must be a dictionary")

        if analysis_type not in SUPPORTED_ANALYSIS_TYPES:
            raise ValueError(f"Invalid analysis type: {analysis_type}. Supported types: {SUPPORTED_ANALYSIS_TYPES}")

        # Validate each document has content
        for doc_name, doc_data in documents.items():
            if not isinstance(doc_data, dict):
                raise ValueError(f"Document '{doc_name}' must be a dictionary")
            if 'content' not in doc_data:
                raise ValueError(f"Document '{doc_name}' must contain a 'content' key")
            if not doc_data['content']:
                raise ValueError(f"Document '{doc_name}' content cannot be empty")

        # Prepare context from documents
        context_docs = ResponseParser.prepare_context_documents(documents)

        # Create analysis prompt based on type
        prompt = self._get_analysis_prompt(analysis_type, context_docs, strategy_text, checklist_results)

        return self.generate_text(prompt, max_length=3000)

    def _get_analysis_prompt(self, analysis_type: str, context_docs: List[str],
                           strategy_text: Optional[str] = None,
                           checklist_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the appropriate analysis prompt based on analysis type.

        Args:
            analysis_type: Type of analysis to perform
            context_docs: Prepared context documents
            strategy_text: Optional strategy document content
            checklist_results: Optional existing checklist results

        Returns:
            Generated prompt for the specified analysis type

        Raises:
            ValueError: If analysis_type is not supported
        """
        if analysis_type == "overview":
            return ResponseParser.create_overview_prompt(context_docs, strategy_text, checklist_results)
        if analysis_type == "strategic":
            return ResponseParser.create_strategic_prompt(context_docs, strategy_text, checklist_results)
        if analysis_type == "checklist":
            return ResponseParser.create_checklist_prompt(context_docs)
        if analysis_type == "questions":
            return ResponseParser.create_questions_prompt(context_docs)

        raise ValueError(f"Unknown analysis type: {analysis_type}")


    # Removed decorator to avoid circular imports
    def answer_question(
        self,
        question: str,
        context_docs: List[str],
        max_length: Optional[int] = None
    ) -> str:
        """
        Answer a specific question using AI with document context.

        This method performs question answering by analyzing the provided
        question against relevant document excerpts to provide accurate,
        context-aware answers.

        Args:
            question: The question to answer. Should be clear and specific
                     for best results (e.g., "What is the company's revenue?"
                     rather than "Tell me about revenue").
            context_docs: List of relevant document excerpts that may contain
                         information to answer the question. Should be
                         pre-filtered to most relevant content.
            max_length: Optional maximum length for the answer in characters.
                      If None, uses service default (typically 2000 chars).

        Returns:
            AI-generated answer with citations and context where applicable

        Raises:
            AIError: If question answering fails or service is unavailable

        Example:
            context = [
                "The company reported $50M in revenue for Q4 2023...",
                "Revenue growth was 15% compared to previous year..."
            ]
            answer = ai_service.answer_question(
                "What was the company's revenue for Q4 2023?",
                context
            )
        """
        # Input validation
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        if not context_docs:
            raise ValueError("Context documents list cannot be None or empty")

        if not isinstance(context_docs, list):
            raise ValueError("Context documents must be a list")

        # Validate each context document
        for i, doc in enumerate(context_docs):
            if not isinstance(doc, str):
                raise ValueError(f"Context document at index {i} must be a string")
            if not doc.strip():
                raise ValueError(f"Context document at index {i} cannot be empty or whitespace only")

        prompt = ResponseParser.create_question_answer_prompt(question, context_docs)
        return self.generate_text(prompt, max_length=max_length or QA_MAX_TOKENS)

    def get_token_usage_estimate(self, text: str) -> int:
        """
        Estimate token usage for a given text using character-based approximation.

        This method provides a rough estimate of token count based on character
        length. Actual token counts may vary depending on the specific tokenizer
        used by the AI model.

        Args:
            text: Text to estimate token count for. Can be any string content
                 including prompts, documents, or responses.

        Returns:
            Estimated token count (integer). Uses approximation of ~4 characters
            per token, which is typical for English text with Claude models.

        Note:
            This is an approximation. For precise token counting, use the
            actual tokenizer for the specific AI model being used.

        Example:
            estimate = ai_service.get_token_usage_estimate("Hello, how are you?")
            # Returns approximately 5-6 tokens
        """
        if not text:
            return 0

        # Rough estimation: ~4 characters per token for English text
        # This is a conservative estimate that works well for Claude models
        return len(text) // 4

    def is_within_token_limit(self, text: str, max_tokens: int = 100000) -> bool:
        """
        Check if text is within specified token limits.

        This method helps prevent token overflow by checking if the estimated
        token count for a given text is within acceptable limits.

        Args:
            text: Text to check for token limit compliance
            max_tokens: Maximum allowed tokens. Default is 100,000 which is
                       a conservative limit for most AI models.

        Returns:
            True if estimated token count is within the specified limit,
            False if it exceeds the limit.

        Note:
            Uses character-based estimation which may not be perfectly accurate.
            For critical token limit checking, consider using the actual tokenizer.

        Example:
            if ai_service.is_within_token_limit(document_content, 8000):
                # Safe to process
                analysis = ai_service.analyze_documents(docs, "overview")
            else:
                # Need to truncate or split content
                print("Content too long for processing")
        """
        if not text:
            return True

        estimated_tokens = self.get_token_usage_estimate(text)
        return estimated_tokens <= max_tokens


# Factory function for easy service creation
def create_ai_service(
    api_key: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 4000
) -> AIService:
    """
    Create and configure an AI service instance with the given parameters.

    This factory function provides a convenient way to create AI service instances
    with proper configuration and validation. It handles all the setup steps
    including configuration validation and service initialization.

    Args:
        api_key: Anthropic API key for authentication. Must be a valid
                Anthropic API key with sufficient permissions.
        model: Claude model to use for AI operations. Examples:
              - "claude-3-5-sonnet" (recommended for most use cases)
              - "claude-3-5-haiku-20241022" (faster, less expensive)
              - "claude-3-opus-20240229" (most capable, more expensive)
        temperature: Sampling temperature for response generation (0.0 to 1.0).
                   Lower values (0.1) produce more deterministic responses.
                   Higher values (0.7+) produce more creative responses.
        max_tokens: Maximum tokens to generate in AI responses.
                   Default 4000 tokens provides good balance of length and cost.

    Returns:
        Fully configured and validated AIService instance ready for use

    Raises:
        ConfigError: If configuration parameters are invalid
        AIError: If AI service initialization fails

    Example:
        # Basic usage
        ai_service = create_ai_service("sk-ant-...", "claude-3-5-sonnet")

        # Advanced configuration
        ai_service = create_ai_service(
            api_key="sk-ant-...",
            model="claude-3-5-haiku-20241022",
            temperature=0.2,
            max_tokens=QA_MAX_TOKENS
        )

        # Use the service
        if ai_service.is_available:
            answer = ai_service.answer_question("What is AI?", ["AI is artificial intelligence..."])
    """
    # Validate and resolve API key
    api_key = _resolve_api_key(api_key)

    config = AIConfig(
        api_key=api_key, 
        model=model, 
        temperature=temperature, 
        max_tokens=max_tokens
    )
    return AIService(config)


def _resolve_api_key(api_key: Optional[str]) -> str:
    """
    Resolve API key from parameter or environment variable.

    Args:
        api_key: API key provided by user, or None

    Returns:
        Resolved API key string

    Raises:
        ConfigError: If no API key is available
    """
    if api_key is not None:
        return api_key

    import os
    env_key = os.getenv('ANTHROPIC_API_KEY')
    if env_key is not None:
        return env_key

    raise create_config_error(
        "AI API key is missing",
        recovery_hint="Please set ANTHROPIC_API_KEY environment variable or pass api_key parameter"
    )
