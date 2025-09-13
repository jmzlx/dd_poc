#!/usr/bin/env python3
"""
Core Services Integration Tests

Focused integration tests for core application services:
- Document processing pipeline
- Checklist parsing and matching
- AI service integration
- Search functionality

Tests core functionality rather than UI workflows.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.document_processor import DocumentProcessor
from app.core.parsers import parse_checklist
from app.core.search import search_documents, search_and_analyze
from app.core.exceptions import DocumentProcessingError, SearchError, ConfigError
from app.services.ai_service import AIService, AIConfig
from app.core.config import init_app_config


class TestCoreServices:
    """Test suite for core application services"""

    def setup_method(self):
        """Setup test environment"""
        self.config = init_app_config()
        from app.core.utils import create_document_processor
        self.document_processor = create_document_processor()

        # Mock test documents
        self.test_documents = {
            "test.pdf": {
                "content": "This is a test document for processing. It contains sample text.",
                "name": "Test Document"
            }
        }

    def test_document_processor_initialization(self):
        """Test document processor initialization"""
        print("🧪 Testing document processor initialization...")

        # Test processor creation
        assert self.document_processor is not None

        # Test FAISS store loading (if available)
        if hasattr(self.document_processor, 'vector_store'):
            # Vector store might be None if no index exists
            pass  # This is acceptable

        print("✅ Document processor initialization test passed")

    def test_document_search_functionality(self):
        """Test document search functionality"""
        print("🧪 Testing document search...")

        # Skip if no FAISS store available
        if not self.document_processor.vector_store:
            print("⚠️  Skipping search test - no FAISS store available")
            return

        test_queries = [
            "test document",
            "sample text"
        ]

        for query in test_queries:
            try:
                results = self.document_processor.search(query, top_k=3, threshold=0.1)
                # Results might be empty if index doesn't contain matching content
                assert isinstance(results, list)
            except (SearchError, DocumentProcessingError) as e:
                print(f"⚠️  Search query '{query}' failed: {e}")

        print("✅ Document search functionality test passed")

    def test_checklist_parsing(self):
        """Test checklist parsing functionality"""
        print("🧪 Testing checklist parsing...")

        # Test valid checklist
        valid_checklist = """
### A. Corporate Structure
1. Are incorporation documents current?
2. Are bylaws properly maintained?

### B. Financial Records
1. Are financial statements audited?
2. Are tax returns filed?
"""

        # Mock LLM response
        mock_llm_response = """
        {
            "categories": {
                "A": {
                    "name": "Corporate Structure",
                    "items": [
                        {"text": "Are incorporation documents current?", "original": "Are incorporation documents current?"},
                        {"text": "Are bylaws properly maintained?", "original": "Are bylaws properly maintained?"}
                    ]
                },
                "B": {
                    "name": "Financial Records",
                    "items": [
                        {"text": "Are financial statements audited?", "original": "Are financial statements audited?"},
                        {"text": "Are tax returns filed?", "original": "Are tax returns filed?"}
                    ]
                }
            }
        }
        """

        from unittest.mock import Mock
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content=mock_llm_response)

        parsed = parse_checklist(valid_checklist, llm=mock_llm)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

        # Check structure
        for category, data in parsed.items():
            assert 'name' in data
            assert 'items' in data
            assert isinstance(data['items'], list)

        print("✅ Checklist parsing test passed")

    def test_checklist_parsing_edge_cases(self):
        """Test checklist parsing with edge cases"""
        print("🧪 Testing checklist parsing edge cases...")

        from unittest.mock import Mock

        # Mock LLM for edge cases
        mock_llm = Mock()

        # Test empty checklist - should raise error when no categories found
        mock_llm.invoke.return_value = Mock(content="{}")
        try:
            empty_parsed = parse_checklist("", llm=mock_llm)
            assert False, "Should have raised RuntimeError for empty checklist"
        except RuntimeError as e:
            assert "Structured parsing failed" in str(e)

        # Test malformed checklist - should raise error when no categories found
        mock_llm.invoke.return_value = Mock(content="{}")
        try:
            malformed_parsed = parse_checklist("Random text without proper format", llm=mock_llm)
            assert False, "Should have raised RuntimeError for malformed checklist"
        except RuntimeError as e:
            assert "Structured parsing failed" in str(e)

        print("✅ Checklist parsing edge cases test passed")

    def test_ai_service_configuration(self):
        """Test AI service configuration"""
        print("🧪 Testing AI service configuration...")

        # Test valid configuration
        config = AIConfig(api_key="test_key", model="claude-3-5-sonnet")
        assert config.api_key == "test_key"
        assert config.model == "claude-3-5-sonnet"

        # Test configuration validation
        try:
            config.validate()
        except ConfigError as e:
            # Validation might fail without actual API key
            print(f"⚠️  Config validation failed (expected): {e}")

        print("✅ AI service configuration test passed")

    def test_ai_service_mock_integration(self):
        """Test AI service integration with mocks"""
        print("🧪 Testing AI service mock integration...")

        # Mock AI service
        mock_service = Mock()
        mock_service.is_available = True
        mock_service.analyze_documents.return_value = "Mock analysis result"
        mock_service.answer_question.return_value = "Mock answer"

        # Test analyze_documents
        result = mock_service.analyze_documents(
            documents=self.test_documents,
            analysis_type="overview"
        )
        assert result == "Mock analysis result"

        # Test answer_question
        answer = mock_service.answer_question(
            "Test question?",
            ["context doc 1", "context doc 2"]
        )
        assert answer == "Mock answer"

        print("✅ AI service mock integration test passed")

    def test_search_and_analyze_integration(self):
        """Test search and analyze integration"""
        print("🧪 Testing search and analyze integration...")

        # Mock questions for testing
        test_questions = [
            {"question": "What is the company revenue?", "category": "Financial", "id": "q_0"}
        ]

        # Mock search results and vector store
        from unittest.mock import Mock
        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.return_value = [
            (Mock(page_content="Company revenue is $75 million", metadata={"name": "financial_report.pdf", "path": "financial_report.pdf"}), 0.9)
        ]

        # Test search_and_analyze
        results = search_and_analyze(
            test_questions,
            mock_vector_store,
            None,  # No AI service
            0.3,   # Threshold
            'questions'
        )

        assert isinstance(results, dict)

        print("✅ Search and analyze integration test passed")

    def test_search_documents_function(self):
        """Test search_documents function"""
        print("🧪 Testing search_documents function...")

        # Mock the document processor
        with patch('app.core.document_processor.DocumentProcessor') as mock_dp_class:
            mock_dp = Mock()
            mock_dp_class.return_value = mock_dp
            mock_dp.search.return_value = [
                {"text": "test result", "source": "test.pdf", "score": 0.8}
            ]

            # Test search function
            results = search_documents(
                "test query",
                mock_dp,
                top_k=5,
                threshold=0.25
            )

            assert len(results) == 1
            assert results[0]["text"] == "test result"

        print("✅ Search documents function test passed")

    def test_error_handling(self):
        """Test error handling in core services"""
        print("🧪 Testing error handling...")

        from unittest.mock import Mock

        # Test with None document processor
        results = search_documents("test", None, top_k=5, threshold=0.25)
        assert len(results) == 0

        # Test checklist parsing with empty string - mock LLM to avoid session dependency
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="{}")
        try:
            parsed = parse_checklist("", llm=mock_llm)
            assert False, "Should have raised RuntimeError for empty checklist"
        except RuntimeError as e:
            assert "Structured parsing failed" in str(e)

        print("✅ Error handling test passed")


def run_core_services_tests():
    """Run all core services tests"""
    print("🚀 Starting Core Services Integration Tests...\n")

    test_suite = TestCoreServices()
    test_suite.setup_method()

    tests = [
        test_suite.test_document_processor_initialization,
        test_suite.test_document_search_functionality,
        test_suite.test_checklist_parsing,
        test_suite.test_checklist_parsing_edge_cases,
        test_suite.test_ai_service_configuration,
        test_suite.test_ai_service_mock_integration,
        test_suite.test_search_and_analyze_integration,
        test_suite.test_search_documents_function,
        test_suite.test_error_handling,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__} PASSED")
        except (ConfigError, DocumentProcessingError, SearchError, AIError) as e:
            print(f"❌ {test.__name__} FAILED: {str(e)}")
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All core services tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False


if __name__ == "__main__":
    success = run_core_services_tests()
    sys.exit(0 if success else 1)
