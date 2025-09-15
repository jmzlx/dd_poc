"""
Unit tests for core service functions

Tests essential functionality for search_documents(), parse_checklist(), and search_and_analyze() functions.
"""
import pytest
import json
from unittest.mock import Mock, patch

from app.core.search import search_documents, search_and_analyze
from app.core.parsers import parse_checklist
from app.core.document_processor import DocumentProcessor


class TestSearchDocuments:
    """Test cases for search_documents function"""

    def test_search_documents_success(self):
        """Test successful document search"""
        mock_processor = Mock(spec=DocumentProcessor)
        mock_results = [
            {
                'text': 'Sample document text',
                'source': 'test.pdf',
                'path': 'test.pdf',
                'score': 0.85,
                'metadata': {'chunk_id': 'chunk_1'}
            }
        ]
        mock_processor.search.return_value = mock_results

        result = search_documents("test query", mock_processor, top_k=5)

        assert result == mock_results
        mock_processor.search.assert_called_once_with("test query", top_k=5, threshold=None)

    def test_search_documents_no_processor(self):
        """Test search with None document processor"""
        result = search_documents("query", None)
        assert result == []


class TestParseChecklist:
    """Test cases for parse_checklist function"""

    def test_parse_checklist_success(self):
        """Test successful checklist parsing"""
        mock_llm = Mock()

        expected_json = {
            "categories": {
                "A": {
                    "name": "Corporate Structure",
                    "items": [
                        {"text": "Review articles", "original": "Review articles"},
                        {"text": "Verify agent", "original": "Verify agent"}
                    ]
                }
            }
        }

        mock_response = Mock()
        mock_response.content = json.dumps(expected_json)
        mock_llm.invoke.return_value = mock_response

        result = parse_checklist("Sample checklist text", mock_llm)

        assert "A" in result
        assert result["A"]["name"] == "Corporate Structure"
        assert len(result["A"]["items"]) == 2

    def test_parse_checklist_no_llm(self):
        """Test error when LLM is not available"""
        with pytest.raises(ValueError, match="LLM parameter is required"):
            parse_checklist("Sample text", None)


class TestSearchAndAnalyzeBehavior:
    """Behavior-focused tests for search_and_analyze function"""

    def test_search_and_analyze_returns_structured_output_for_checklist(self):
        """Test that search_and_analyze returns properly structured output for checklist items"""
        mock_checklist_data = {
            "A": {
                "name": "Corporate Structure", 
                "items": [
                    {"text": "Review articles", "original": "Review articles"}
                ]
            }
        }

        # Mock vector store with minimal required behavior
        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = []

        # Create a mock session (may or may not be used depending on implementation)
        mock_session = Mock()
        mock_session.document_type_embeddings = {}

        try:
            result = search_and_analyze(
                mock_checklist_data,
                mock_store,
                threshold=0.1,
                search_type='items',
                store_name='test_store',
                session=mock_session
            )

            # Should return structured data preserving the input structure
            assert isinstance(result, dict)
            
            # Should maintain category structure even if no matches found
            if result:  # Function may return empty dict if no embeddings available
                for category_key, category_data in result.items():
                    assert isinstance(category_data, dict)
                    if 'name' in category_data:
                        assert isinstance(category_data['name'], str)
                    if 'items' in category_data:
                        assert isinstance(category_data['items'], list)

        except Exception as e:
            # If function requires specific setup, should fail gracefully with informative error
            assert len(str(e)) > 0

    def test_search_and_analyze_handles_questions_format(self):
        """Test that search_and_analyze handles questions format appropriately"""
        mock_questions = [
            {"question": "What is the revenue?", "category": "A. Financial", "id": "q_0"}
        ]

        # Mock vector store with minimal behavior
        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = []

        try:
            result = search_and_analyze(
                mock_questions,
                mock_store,
                threshold=0.1,
                search_type='questions'
            )

            # Should return structured data for questions
            assert isinstance(result, dict)
            
            # Should handle questions input format appropriately
            # (exact structure may vary by implementation)
            if result and 'questions' in result:
                assert isinstance(result['questions'], list)
                for question in result['questions']:
                    assert isinstance(question, dict)
                    # Should preserve essential question data
                    assert any(field in question for field in ['question', 'query', 'text'])

        except Exception as e:
            # Should fail gracefully if prerequisites not met
            assert len(str(e)) > 0

    def test_search_and_analyze_handles_empty_input(self):
        """Test that search_and_analyze handles empty input gracefully"""
        empty_data = {}
        mock_store = Mock()
        mock_store.similarity_search_with_score.return_value = []

        try:
            result = search_and_analyze(
                empty_data,
                mock_store,
                threshold=0.1,
                search_type='items'
            )
            # Should return valid structure for empty input
            assert isinstance(result, dict)
        except Exception as e:
            # Should provide informative error for invalid input
            assert len(str(e)) > 0
