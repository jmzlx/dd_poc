#!/usr/bin/env python3
"""
AI Handler

Handles AI operations and coordinates between UI and AI service.
"""

from typing import Optional, List

from app.ui.session_manager import SessionManager
from app.services.ai_service import AIService, create_ai_service
from app.core.exceptions import AIError, ConfigError, create_ai_error
from app.ui.error_handler import handle_processing_errors
from app.core.logging import logger


class AIHandler:
    """
    AI handler that manages AI operations using the AI service.

    Provides a clean interface between UI and AI service.
    """

    def __init__(self, session: SessionManager):
        """Initialize handler with session manager"""
        self.session = session
        self._ai_service: Optional[AIService] = None

    @handle_processing_errors("AI service setup", "Please check your API key and try again")
    def setup_agent(self, api_key: str, model_choice: str) -> bool:
        """
        Setup AI service with given credentials.

        Args:
            api_key: Anthropic API key
            model_choice: Claude model to use

        Returns:
            True if AI service was successfully initialized

        Raises:
            AIError: If AI service setup fails
            ConfigError: If API key or model is invalid
        """
        # Get appropriate max_tokens for the model
        from app.core.config import get_app_config
        config = get_app_config()
        
        # Adjust max_tokens based on model limitations
        max_tokens = config.model['max_tokens']
        original_max_tokens = max_tokens
        
        if 'haiku' in model_choice.lower():
            # Claude Haiku has a maximum of 8192 output tokens
            max_tokens = min(max_tokens, 8192)
        elif 'sonnet' in model_choice.lower():
            # Claude Sonnet models can handle higher token counts
            max_tokens = min(max_tokens, 8192)  # Conservative limit for reliability
        
        if max_tokens != original_max_tokens:
            logger.info(f"Adjusted max_tokens for {model_choice}: {original_max_tokens} -> {max_tokens}")
        
        logger.info(f"Initializing AI service: model={model_choice}, max_tokens={max_tokens}, temperature={config.model['temperature']}")
        
        # Create AI service with proper token limits
        self._ai_service = create_ai_service(
            api_key=api_key, 
            model=model_choice,
            temperature=config.model['temperature'],
            max_tokens=max_tokens
        )

        # Check if service was created successfully
        if self._ai_service is None:
            raise create_ai_error(
                "AI service creation failed",
                recovery_hint="Please check your API key and try again"
            )

        # Test the service
        if self._ai_service.is_available:
            # Store the AI service in the session for other components to access
            self.session.agent = self._ai_service
            return True
        else:
            raise create_ai_error(
                "AI service initialization failed",
                recovery_hint="Please check your API key and network connection"
            )

    def is_agent_available(self) -> bool:
        """
        Check if AI service is available and ready.

        Returns:
            True if AI service is available
        """
        # Check local AI service first
        if self._ai_service is not None and self._ai_service.is_available:
            return True
        
        # Check session for existing agent
        if self.session.agent is not None:
            # Update local reference if session has an agent
            self._ai_service = self.session.agent
            return self._ai_service.is_available
        
        return False


    @handle_processing_errors("Report generation", "Please check your documents and try again")
    def generate_report(self, report_type: str, **kwargs) -> Optional[str]:
        """
        Generate a report using the AI service.

        Args:
            report_type: Type of report ('overview', 'strategic', 'checklist', 'questions')
            **kwargs: Additional arguments for report generation

        Returns:
            Generated report content or None if failed

        Raises:
            AIError: If report generation fails
        """
        if not self.is_agent_available():
            raise create_ai_error(
                "AI service not available",
                recovery_hint="Please configure your API key in the sidebar"
            )

        documents = kwargs.get('documents', {})
        strategy_text = kwargs.get('strategy_text')
        checklist_results = kwargs.get('checklist_results')

        return self._ai_service.analyze_documents(
            documents=documents,
            analysis_type=report_type,
            strategy_text=strategy_text,
            checklist_results=checklist_results
        )


    @handle_processing_errors("Question answering", "Please try rephrasing your question")
    def answer_question(self, question: str, context_docs: List[str]) -> str:
        """
        Answer a specific question using AI.

        Args:
            question: The question to answer
            context_docs: List of relevant document excerpts

        Returns:
            AI-generated answer

        Raises:
            AIError: If question answering fails
        """
        if not self.is_agent_available():
            raise create_ai_error(
                "AI service not available",
                recovery_hint="Please configure your API key in the sidebar"
            )

        return self._ai_service.answer_question(question, context_docs)

    @property
    def llm(self):
        """Get the underlying LLM instance"""
        # Check local AI service first
        if self._ai_service is not None:
            return self._ai_service.llm
        
        # Check session for existing agent
        if self.session.agent is not None:
            # Update local reference if session has an agent
            self._ai_service = self.session.agent
            return self._ai_service.llm
        
        return None
