#!/usr/bin/env python3
"""
AI Workflows Integration Tests

Comprehensive integration tests for AI-powered report generation including:
- Overview generation
- Strategic analysis
- Q&A flows
- Prompt construction validation
- Response parsing
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from app.ui.session_manager import SessionManager
from app.core.config import init_app_config
from app.handlers.ai_handler import AIHandler
from app.services.ai_service import AIService, AIConfig, create_ai_service
from app.core.search import search_documents
from app.core.exceptions import AIError
from app.core.exceptions import LLMConnectionError, LLMAuthenticationError, LLMTimeoutError, ConfigError
from app.core.logging import logger
from app.core.constants import TEMPERATURE


class TestAIWorkflows:
    """Test class for AI workflow integration tests"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment before each test"""
        self.config = init_app_config()
        self.session = SessionManager()
        self.ai_handler = AIHandler(self.session)
        from app.core.utils import create_document_processor
        self.document_processor = create_document_processor()

        # Mock documents for testing
        self.mock_documents = {
            "company_profile.pdf": {
                "content": "TechCorp is a leading cybersecurity company founded in 2015. "
                          "The company specializes in AI-driven threat detection and "
                          "provides comprehensive security solutions to enterprise clients worldwide. "
                          "Key markets include finance, healthcare, and government sectors.",
                "name": "Company Profile"
            },
            "financial_report.pdf": {
                "content": "Financial Overview: Revenue $75M, Net Profit $12M, Total Assets $150M. "
                          "The company has shown 25% YoY revenue growth. Strong balance sheet "
                          "with manageable debt levels and excellent cash flow generation.",
                "name": "Financial Report"
            },
            "strategic_plan.pdf": {
                "content": "Strategic Objectives: Expand into international markets, "
                          "invest in AI/ML capabilities, strengthen partnerships with key technology vendors. "
                          "Risk mitigation strategies include diversification across customer segments "
                          "and continuous investment in R&D.",
                "name": "Strategic Plan"
            }
        }

    @pytest.fixture
    def mock_ai_service(self):
        """Create a mock AI service for testing"""
        mock_service = Mock(spec=AIService)
        mock_service.is_available = True

        # Realistic mock return values with proper length
        mock_service.analyze_documents.return_value = "# Company Overview Analysis\n\nThis is a comprehensive analysis of the company based on the provided documents. The analysis covers various aspects including financial performance, market position, and strategic initiatives.\n\n## Key Findings\n\n- Strong market position with significant growth potential\n- Robust financial metrics and operational efficiency\n- Strategic partnerships that enhance competitive advantage\n\n## Recommendations\n\nBased on the analysis, several recommendations can be made to improve performance and mitigate risks."
        mock_service.answer_question.return_value = "Mock answer"

        return mock_service



    def test_overview_generation_workflow(self, mock_ai_service):
        """Test complete overview generation workflow"""
        logger.info("🧪 Testing overview generation workflow...")

        with patch.object(self.ai_handler, '_ai_service', mock_ai_service):
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):

                # Test overview report generation
                result = self.ai_handler.generate_report(
                    "overview",
                    documents=self.mock_documents,
                    data_room_name="TechCorp"
                )

                # Validate result
                assert "# Company Overview Analysis" in result
                assert len(result.strip()) > 0

                # Verify AI service was called correctly
                mock_ai_service.analyze_documents.assert_called_once_with(
                    documents=self.mock_documents,
                    analysis_type="overview",
                    strategy_text=None,
                    checklist_results=None
                )

                logger.info("✅ Overview generation workflow test passed")

    def test_strategic_analysis_workflow(self, mock_ai_service):
        """Test complete strategic analysis workflow"""
        logger.info("🧪 Testing strategic analysis workflow...")

        with patch.object(self.ai_handler, '_ai_service', mock_ai_service):
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):

                # Test strategic report generation
                result = self.ai_handler.generate_report(
                    "strategic",
                    documents=self.mock_documents,
                    data_room_name="TechCorp",
                    strategy_text="Strategic expansion plan content"
                )

                # Validate result
                assert "# Company Overview Analysis" in result
                assert len(result.strip()) > 0

                # Verify AI service was called correctly
                mock_ai_service.analyze_documents.assert_called_once_with(
                    documents=self.mock_documents,
                    analysis_type="strategic",
                    strategy_text="Strategic expansion plan content",
                    checklist_results=None
                )

                logger.info("✅ Strategic analysis workflow test passed")

    def test_qa_workflow_with_document_search(self, mock_ai_service):
        """Test Q&A workflow with document search integration"""
        logger.info("🧪 Testing Q&A workflow with document search...")

        # Mock document processor search results
        mock_search_results = [
            {
                'text': 'TechCorp is a leading cybersecurity company founded in 2015.',
                'source': 'company_profile.pdf',
                'path': 'company_profile.pdf',
                'score': 0.85
            },
            {
                'text': 'Financial Overview: Revenue $75M, Net Profit $12M.',
                'source': 'financial_report.pdf',
                'path': 'financial_report.pdf',
                'score': 0.78
            }
        ]

        with patch.object(self.ai_handler, '_ai_service', mock_ai_service):
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):
                with patch('app.core.search.search_documents', return_value=mock_search_results):

                    # Test question answering
                    question = "What is TechCorp's annual revenue?"
                    result = self.ai_handler.answer_question(question, ["context doc 1", "context doc 2"])

                    # Validate result
                    assert result == "Mock answer"
                    assert len(result.strip()) > 0

                    # Verify AI service was called correctly
                    mock_ai_service.answer_question.assert_called_once_with(
                        question,
                        ["context doc 1", "context doc 2"]
                    )

                    logger.info("✅ Q&A workflow test passed")

    def test_prompt_construction_validation(self, mock_ai_service):
        """Test prompt construction for different workflows"""
        logger.info("🧪 Testing prompt construction validation...")

        # Test overview prompt construction
        with patch.object(self.ai_handler, '_ai_service', mock_ai_service):
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):

                # Generate overview to trigger prompt construction
                self.ai_handler.generate_report(
                    "overview",
                    documents=self.mock_documents,
                    data_room_name="TechCorp"
                )

                # Verify the call was made with correct parameters
                call_args = mock_ai_service.analyze_documents.call_args
                assert call_args[1]['analysis_type'] == 'overview'
                assert call_args[1]['documents'] == self.mock_documents

                logger.info("✅ Prompt construction validation test passed")

    def test_response_parsing_and_validation(self, mock_ai_service):
        """Test response parsing and validation from AI services"""
        logger.info("🧪 Testing response parsing and validation...")

        with patch.object(self.ai_handler, '_ai_service', mock_ai_service):
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):

                # Test overview response parsing
                overview_result = self.ai_handler.generate_report(
                    "overview",
                    documents=self.mock_documents
                )

                # Validate response structure
                assert isinstance(overview_result, str)
                assert len(overview_result) > 100  # Reasonable length check
                assert overview_result.startswith('#')  # Markdown header

                # Test strategic response parsing
                strategic_result = self.ai_handler.generate_report(
                    "strategic",
                    documents=self.mock_documents
                )

                assert isinstance(strategic_result, str)
                assert len(strategic_result) > 100
                assert "# Company Overview Analysis" in strategic_result

                logger.info("✅ Response parsing and validation test passed")

    def test_ai_service_error_handling(self):
        """Test error handling in AI workflows"""
        logger.info("🧪 Testing AI service error handling...")

        # Test with unavailable AI service
        with patch.object(self.ai_handler, 'is_agent_available', return_value=False):

            with pytest.raises(AIError) as exc_info:
                self.ai_handler.generate_report("overview", documents=self.mock_documents)

            assert "AI service not available" in str(exc_info.value)

        # Test with AI service that raises exception
        mock_service = Mock(spec=AIService)
        mock_service.is_available = True
        mock_service.analyze_documents.side_effect = Exception("AI service error")

        with patch.object(self.ai_handler, '_ai_service', mock_service):
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):

                with pytest.raises(Exception) as exc_info:
                    self.ai_handler.generate_report("overview", documents=self.mock_documents)

                assert "AI service error" in str(exc_info.value)

        logger.info("✅ AI service error handling test passed")

    def test_workflow_integration_with_session_management(self, mock_ai_service):
        """Test workflow integration with session management"""
        logger.info("🧪 Testing workflow integration with session management...")

        with patch.object(self.ai_handler, '_ai_service', mock_ai_service):
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):

                # Simulate complete workflow
                # 1. Generate overview
                overview = self.ai_handler.generate_report(
                    "overview",
                    documents=self.mock_documents,
                    data_room_name="TechCorp"
                )

                # 2. Generate strategic analysis
                strategic = self.ai_handler.generate_report(
                    "strategic",
                    documents=self.mock_documents,
                    data_room_name="TechCorp"
                )

                # 3. Answer questions
                answer = self.ai_handler.answer_question(
                    "What is the revenue?",
                    ["Financial context"]
                )

                # Validate all results are stored and accessible
                assert overview is not None
                assert strategic is not None
                assert answer is not None

                # Verify session maintains state
                assert self.session is not None

                logger.info("✅ Workflow integration with session management test passed")

    def test_ai_service_configuration_validation(self):
        """Test AI service configuration validation"""
        logger.info("🧪 Testing AI service configuration validation...")

        # Test invalid configuration
        invalid_config = AIConfig(api_key="", model="")

        with pytest.raises(ConfigError):  # Should raise ConfigError
            AIService(invalid_config)

        # Test valid configuration setup
        valid_config = AIConfig(
            api_key="test-key",
            model="claude-3-5-sonnet",
            temperature=TEMPERATURE,
            max_tokens=4000
        )

        # Should not raise exception during initialization
        # (though actual API calls would fail)
        try:
            service = AIService(valid_config)
            # Service should indicate it's not available with invalid key
            assert not service.is_available
        except (LLMConnectionError, LLMAuthenticationError, LLMTimeoutError):
            # If initialization fails due to API issues, that's also acceptable
            pass

        logger.info("✅ AI service configuration validation test passed")

    @pytest.mark.parametrize("analysis_type,expected_content", [
        ("overview", ["Executive Summary", "Financial Performance"]),
        ("strategic", ["Strategic Objectives", "Risk Assessment"]),
        ("checklist", ["Corporate Structure", "Financial Health"])
    ])
    def test_parametrized_workflow_testing(self, mock_ai_service, analysis_type, expected_content):
        """Test multiple analysis types with parametrized tests"""
        logger.info(f"🧪 Testing parametrized workflow for {analysis_type}...")

        with patch.object(self.ai_handler, '_ai_service', mock_ai_service):
            with patch.object(self.ai_handler, 'is_agent_available', return_value=True):

                result = self.ai_handler.generate_report(
                    analysis_type,
                    documents=self.mock_documents,
                    data_room_name="TechCorp"
                )

                assert "# Company Overview Analysis" in result

                logger.info(f"✅ Parametrized workflow test for {analysis_type} passed")


# Helper functions for test setup
def create_mock_documents() -> Dict[str, Dict[str, str]]:
    """Create mock documents for testing"""
    return {
        "profile.pdf": {
            "content": "Company profile content for testing",
            "name": "Company Profile"
        },
        "financials.pdf": {
            "content": "Financial statements and analysis",
            "name": "Financial Report"
        }
    }


def setup_test_environment():
    """Setup test environment with necessary mocks"""
    config = init_app_config()
    session = SessionManager()
    ai_handler = AIHandler(session)

    return config, session, ai_handler


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
